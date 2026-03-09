/**
 * ThermoMechSim3D - Thermal-Mechanical Simulation Engine
 * Ported from CUDA kernels to JavaScript
 * Implements finite-difference heat equation solver and thermal stress computation
 */

class ThermoMechSimulation {
    constructor() {
        this.temperature = null;
        this.newTemperature = null;
        this.stress = null;
        this.powerMap = null;
        this.materialGrid = null;
        this.materialProps = []; // Flat array of material properties

        this.config = null;
        this.preset = null;
        this.running = false;
        this.step = 0;
        this.convergence = Infinity;
        this.criticalRegions = 0;
        this.startTime = 0;

        this.onProgress = null;
        this.onComplete = null;
        this.onStepUpdate = null;
        this._stopRequested = false;
    }

    /**
     * Initialize simulation with given configuration
     */
    initialize(config, presetKey) {
        this.config = { ...config };
        this.preset = PRESETS[presetKey] || PRESETS.hbm_stack;

        const { dimX, dimY, dimZ } = this.config;
        const totalCells = dimX * dimY * dimZ;

        // Allocate arrays
        this.temperature = new Float32Array(totalCells);
        this.newTemperature = new Float32Array(totalCells);
        this.stress = new Float32Array(totalCells);
        this.powerMap = new Float32Array(totalCells);
        this.materialGrid = new Int32Array(totalCells);

        // Initialize temperature field with ambient temperature
        this.temperature.fill(this.config.ambientTemp);
        this.newTemperature.fill(this.config.ambientTemp);
        this.stress.fill(0);

        // Build material properties lookup
        this._buildMaterialProps();

        // Create material grid from stackup
        this._createMaterialGrid();

        // Generate power map
        this._generatePowerMap();

        this.step = 0;
        this.convergence = Infinity;
        this.criticalRegions = 0;
        this._stopRequested = false;
    }

    /**
     * Build flat material properties array for fast lookup
     */
    _buildMaterialProps() {
        this.materialProps = [];
        const matNames = Object.keys(MATERIALS);
        for (let i = 0; i < matNames.length; i++) {
            const mat = MATERIALS[matNames[i]];
            this.materialProps.push({
                conductivity: mat.thermal_conductivity,
                density: mat.density,
                specificHeat: mat.specific_heat,
                youngsModulus: mat.youngs_modulus,
                poissonsRatio: mat.poissons_ratio,
                thermalExpansion: mat.thermal_expansion,
                yieldStrength: mat.yield_strength
            });
        }
    }

    /**
     * Create material grid from stackup layers
     */
    _createMaterialGrid() {
        const { dimX, dimY, dimZ } = this.config;
        const stackup = this.preset.stackup;

        // Calculate total thickness
        let totalThickness = 0;
        for (const layer of stackup) {
            totalThickness += layer.thickness;
        }

        // Assign materials based on z-position
        let zPos = 0;
        for (const layer of stackup) {
            const layerStart = zPos / totalThickness;
            zPos += layer.thickness;
            const layerEnd = zPos / totalThickness;

            let zStart = Math.floor(layerStart * dimZ);
            let zEnd = Math.floor(layerEnd * dimZ);

            // Ensure last layer fills to the top
            if (layer === stackup[stackup.length - 1]) {
                zEnd = dimZ;
            }

            const matIdx = getMaterialIndex(layer.material);
            if (matIdx < 0) continue;

            for (let z = zStart; z < zEnd; z++) {
                for (let y = 0; y < dimY; y++) {
                    for (let x = 0; x < dimX; x++) {
                        this.materialGrid[z * dimX * dimY + y * dimX + x] = matIdx;
                    }
                }
            }
        }
    }

    /**
     * Generate power map with Gaussian distribution for powered layers
     */
    _generatePowerMap() {
        const { dimX, dimY, dimZ } = this.config;
        const totalPower = this.config.totalPower;
        const stackup = this.preset.stackup;

        this.powerMap.fill(0);

        // Calculate total thickness
        let totalThickness = 0;
        for (const layer of stackup) {
            totalThickness += layer.thickness;
        }

        // Generate power for each powered layer
        let zPos = 0;
        for (const layer of stackup) {
            const layerStart = zPos / totalThickness;
            zPos += layer.thickness;
            const layerEnd = zPos / totalThickness;

            if (!layer.hasPower) continue;

            const zStart = Math.floor(layerStart * dimZ);
            const zEnd = Math.max(zStart + 1, Math.floor(layerEnd * dimZ));
            const layerPower = totalPower * layer.powerFraction;

            const centerX = dimX / 2;
            const centerY = dimY / 2;
            const sigmaSq = (dimX * dimY) / 50;

            // Calculate Gaussian distribution
            let total = 0;
            for (let z = zStart; z < zEnd; z++) {
                for (let y = 0; y < dimY; y++) {
                    for (let x = 0; x < dimX; x++) {
                        const dx = x - centerX;
                        const dy = y - centerY;
                        const rSq = dx * dx + dy * dy;
                        const val = Math.exp(-rSq / (2 * sigmaSq));
                        const idx = z * dimX * dimY + y * dimX + x;
                        this.powerMap[idx] = val;
                        total += val;
                    }
                }
            }

            // Normalize to desired power
            if (total > 0) {
                const scale = layerPower / total;
                for (let z = zStart; z < zEnd; z++) {
                    for (let y = 0; y < dimY; y++) {
                        for (let x = 0; x < dimX; x++) {
                            const idx = z * dimX * dimY + y * dimX + x;
                            this.powerMap[idx] *= scale;
                        }
                    }
                }
            }
        }
    }

    /**
     * Run the simulation asynchronously with yielding for UI updates
     */
    async run() {
        this.running = true;
        this._stopRequested = false;
        this.startTime = performance.now();

        const maxSteps = this.config.maxSteps;
        const batchSize = 10; // Steps per batch before yielding to UI

        for (let step = 0; step < maxSteps; step++) {
            if (this._stopRequested) break;

            // Run one timestep
            this._timestep();
            this.step = step + 1;

            // Check convergence periodically
            if (step > 0 && step % 20 === 0) {
                this.convergence = this._checkConvergence();
                if (this.convergence < this.config.convergenceThreshold) {
                    break;
                }
            }

            // Yield to UI every batchSize steps
            if (step % batchSize === 0) {
                // Compute statistics
                const stats = this._computeStats();

                if (this.onStepUpdate) {
                    this.onStepUpdate({
                        step: this.step,
                        maxSteps: maxSteps,
                        maxTemp: stats.maxTemp,
                        minTemp: stats.minTemp,
                        maxStress: stats.maxStress,
                        convergence: this.convergence,
                        elapsed: (performance.now() - this.startTime) / 1000
                    });
                }

                if (this.onProgress) {
                    this.onProgress(this.step / maxSteps);
                }

                // Yield to allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        // Final analysis
        this._identifyCriticalRegions();
        const stats = this._computeStats();
        const failureRisk = this._assessFailureRisk(stats);

        this.running = false;

        if (this.onComplete) {
            this.onComplete({
                step: this.step,
                maxTemp: stats.maxTemp,
                minTemp: stats.minTemp,
                maxStress: stats.maxStress,
                convergence: this.convergence,
                criticalRegions: this.criticalRegions,
                elapsed: (performance.now() - this.startTime) / 1000,
                failureRisk: failureRisk
            });
        }
    }

    /**
     * Stop the simulation
     */
    stop() {
        this._stopRequested = true;
    }

    /**
     * Single timestep: solve heat equation + compute stress
     * Implements the same finite-difference scheme as the CUDA kernel
     */
    _timestep() {
        const { dimX, dimY, dimZ, dt, dx, dy, dz, refTemp } = this.config;
        const temp = this.temperature;
        const newTemp = this.newTemperature;
        const stress = this.stress;
        const power = this.powerMap;
        const matGrid = this.materialGrid;
        const matProps = this.materialProps;

        // Solve heat equation: rho*Cp*(dT/dt) = k*Laplacian(T) + Q
        for (let z = 1; z < dimZ - 1; z++) {
            for (let y = 1; y < dimY - 1; y++) {
                for (let x = 1; x < dimX - 1; x++) {
                    const idx = z * dimX * dimY + y * dimX + x;
                    const matId = matGrid[idx];
                    const mat = matProps[matId];

                    if (!mat) continue;

                    const k = mat.conductivity;
                    const rho = mat.density;
                    const cp = mat.specificHeat;

                    // Second-order finite differences for Laplacian
                    const idxXp = idx + 1;
                    const idxXm = idx - 1;
                    const idxYp = idx + dimX;
                    const idxYm = idx - dimX;
                    const idxZp = idx + dimX * dimY;
                    const idxZm = idx - dimX * dimY;

                    const d2Tdx2 = (temp[idxXp] - 2 * temp[idx] + temp[idxXm]) / (dx * dx);
                    const d2Tdy2 = (temp[idxYp] - 2 * temp[idx] + temp[idxYm]) / (dy * dy);
                    const d2Tdz2 = (temp[idxZp] - 2 * temp[idx] + temp[idxZm]) / (dz * dz);

                    const flux = k * (d2Tdx2 + d2Tdy2 + d2Tdz2) + power[idx];
                    const thermalCapacity = rho * cp;

                    newTemp[idx] = temp[idx] + dt * flux / thermalCapacity;

                    // Compute thermal stress (von Mises)
                    // For a constrained body under thermal load, the thermal stress
                    // is driven by the temperature difference from the reference and
                    // the local temperature gradient (mismatch between neighbors).
                    const E = mat.youngsModulus;
                    const nu = mat.poissonsRatio;
                    const alpha = mat.thermalExpansion;
                    const tempDiff = temp[idx] - refTemp;

                    // Biaxial thermal stress for constrained thin film:
                    // sigma = -E * alpha * deltaT / (1 - nu)
                    // This is the dominant stress in 3D-IC stacks
                    const biaxialStress = E * alpha * Math.abs(tempDiff) / (1 - nu);

                    // Additional stress from temperature gradients (mismatch stress)
                    // Use normalized gradients to capture local non-uniformity
                    const dTdx = (temp[idxXp] - temp[idxXm]) / 2;
                    const dTdy = (temp[idxYp] - temp[idxYm]) / 2;
                    const dTdz = (temp[idxZp] - temp[idxZm]) / 2;
                    const gradMag = Math.sqrt(dTdx * dTdx + dTdy * dTdy + dTdz * dTdz);
                    const gradientStress = E * alpha * gradMag / (1 - nu);

                    // Combined von Mises equivalent stress
                    // Biaxial gives the baseline; gradient adds local variation
                    stress[idx] = biaxialStress + 0.5 * gradientStress;
                }
            }
        }

        // Apply boundary conditions
        this._applyBoundaryConditions(newTemp);

        // Swap buffers
        const tmpRef = this.temperature;
        this.temperature = this.newTemperature;
        this.newTemperature = tmpRef;
    }

    /**
     * Apply boundary conditions
     */
    _applyBoundaryConditions(temp) {
        const { dimX, dimY, dimZ } = this.config;
        const boundary = this.preset.boundary;
        const ambientTemp = this.config.ambientTemp;
        const hConv = this.preset.convectionCoeff;
        const bottomTemp = this.preset.bottomTemp;

        // Bottom boundary (z = 0): constant temperature
        if (boundary.bottom === 'constant_temp') {
            for (let y = 0; y < dimY; y++) {
                for (let x = 0; x < dimX; x++) {
                    temp[y * dimX + x] = bottomTemp;
                }
            }
        }

        // Top boundary (z = dimZ-1): convection
        if (boundary.top === 'convection') {
            const zTop = dimZ - 1;
            const zInner = dimZ - 2;
            for (let y = 0; y < dimY; y++) {
                for (let x = 0; x < dimX; x++) {
                    const topIdx = zTop * dimX * dimY + y * dimX + x;
                    const innerIdx = zInner * dimX * dimY + y * dimX + x;
                    const matId = this.materialGrid[topIdx];
                    const mat = this.materialProps[matId];
                    if (!mat) continue;
                    const k = mat.conductivity;
                    const dzVal = this.config.dz;
                    // Newton's law of cooling at boundary
                    temp[topIdx] = (temp[innerIdx] * k / dzVal + hConv * ambientTemp) / (k / dzVal + hConv);
                }
            }
        }

        // Side boundaries: adiabatic (zero heat flux)
        if (boundary.sides === 'adiabatic') {
            for (let z = 0; z < dimZ; z++) {
                for (let y = 0; y < dimY; y++) {
                    // x = 0
                    const leftIdx = z * dimX * dimY + y * dimX;
                    const leftInner = leftIdx + 1;
                    temp[leftIdx] = temp[leftInner];

                    // x = dimX - 1
                    const rightIdx = z * dimX * dimY + y * dimX + (dimX - 1);
                    const rightInner = rightIdx - 1;
                    temp[rightIdx] = temp[rightInner];
                }
                for (let x = 0; x < dimX; x++) {
                    // y = 0
                    const frontIdx = z * dimX * dimY + x;
                    const frontInner = frontIdx + dimX;
                    temp[frontIdx] = temp[frontInner];

                    // y = dimY - 1
                    const backIdx = z * dimX * dimY + (dimY - 1) * dimX + x;
                    const backInner = backIdx - dimX;
                    temp[backIdx] = temp[backInner];
                }
            }
        }
    }

    /**
     * Check convergence (max temperature change)
     */
    _checkConvergence() {
        let maxDiff = 0;
        const len = this.temperature.length;
        for (let i = 0; i < len; i++) {
            const diff = Math.abs(this.temperature[i] - this.newTemperature[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        return maxDiff;
    }

    /**
     * Compute statistics
     */
    _computeStats() {
        let maxTemp = -Infinity;
        let minTemp = Infinity;
        let maxStress = 0;
        const len = this.temperature.length;

        for (let i = 0; i < len; i++) {
            if (this.temperature[i] > maxTemp) maxTemp = this.temperature[i];
            if (this.temperature[i] < minTemp) minTemp = this.temperature[i];
            if (this.stress[i] > maxStress) maxStress = this.stress[i];
        }

        return { maxTemp, minTemp, maxStress };
    }

    /**
     * Identify critical stress regions
     */
    _identifyCriticalRegions() {
        const threshold = this.config.stressThreshold;
        let count = 0;
        const len = this.stress.length;

        for (let i = 0; i < len; i++) {
            if (this.stress[i] > threshold) {
                count++;
            }
        }

        this.criticalRegions = count;
    }

    /**
     * Assess failure risk based on simulation results
     */
    _assessFailureRisk(stats) {
        const risks = [];

        // TSV stress margin
        const tsvMargin = this._calculateTSVMargin();
        risks.push({
            component: 'TSV Array',
            margin: tsvMargin,
            status: tsvMargin > 0.7 ? 'ok' : tsvMargin > 0.5 ? 'warn' : 'critical'
        });

        // Microbump stress margin
        const mbMargin = this._calculateMicrobumpMargin();
        risks.push({
            component: 'Microbump Interface',
            margin: mbMargin,
            status: mbMargin > 0.7 ? 'ok' : mbMargin > 0.5 ? 'warn' : 'critical'
        });

        // Thermal cycling fatigue
        const tempRange = stats.maxTemp - stats.minTemp;
        const fatigueRisk = tempRange > 80 ? 'critical' : tempRange > 50 ? 'warn' : 'ok';
        risks.push({
            component: 'Thermal Cycling',
            margin: Math.max(0, 1 - tempRange / 100),
            status: fatigueRisk
        });

        // Estimated MTF
        const mtf = this._estimateMTF(stats.maxTemp, stats.maxStress);
        risks.push({
            component: 'Estimated MTF',
            value: mtf.toFixed(1) + ' years',
            status: mtf > 10 ? 'ok' : mtf > 5 ? 'warn' : 'critical'
        });

        return risks;
    }

    _calculateTSVMargin() {
        const { dimX, dimY, dimZ } = this.config;
        const stackup = this.preset.stackup;
        let totalThickness = 0;
        for (const layer of stackup) totalThickness += layer.thickness;

        let maxTSVStress = 0;
        let tsvYield = 50e6; // Cu_TSV yield strength

        let zPos = 0;
        for (const layer of stackup) {
            const layerStart = zPos / totalThickness;
            zPos += layer.thickness;
            const layerEnd = zPos / totalThickness;

            if (layer.material !== 'Cu_TSV') continue;

            const zStart = Math.floor(layerStart * dimZ);
            const zEnd = Math.max(zStart + 1, Math.floor(layerEnd * dimZ));

            for (let z = zStart; z < zEnd; z++) {
                for (let y = 0; y < dimY; y++) {
                    for (let x = 0; x < dimX; x++) {
                        const idx = z * dimX * dimY + y * dimX + x;
                        if (this.stress[idx] > maxTSVStress) {
                            maxTSVStress = this.stress[idx];
                        }
                    }
                }
            }
        }

        return maxTSVStress > 0 ? Math.max(0, 1 - maxTSVStress / tsvYield) : 1.0;
    }

    _calculateMicrobumpMargin() {
        const { dimX, dimY, dimZ } = this.config;
        const stackup = this.preset.stackup;
        let totalThickness = 0;
        for (const layer of stackup) totalThickness += layer.thickness;

        let maxSolderStress = 0;
        const solderYield = 32e6; // SAC305 yield strength

        let zPos = 0;
        for (const layer of stackup) {
            const layerStart = zPos / totalThickness;
            zPos += layer.thickness;
            const layerEnd = zPos / totalThickness;

            if (layer.material !== 'SAC305') continue;

            const zStart = Math.floor(layerStart * dimZ);
            const zEnd = Math.max(zStart + 1, Math.floor(layerEnd * dimZ));

            for (let z = zStart; z < zEnd; z++) {
                for (let y = 0; y < dimY; y++) {
                    for (let x = 0; x < dimX; x++) {
                        const idx = z * dimX * dimY + y * dimX + x;
                        if (this.stress[idx] > maxSolderStress) {
                            maxSolderStress = this.stress[idx];
                        }
                    }
                }
            }
        }

        return maxSolderStress > 0 ? Math.max(0, 1 - maxSolderStress / solderYield) : 1.0;
    }

    _estimateMTF(maxTemp, maxStress) {
        // Black's equation approximation for electromigration MTF
        // MTF = A * J^(-n) * exp(Ea / (kB * T))
        // Simplified: higher temp and stress = lower lifetime
        const kB = 8.617e-5; // Boltzmann constant in eV/K
        const Ea = 0.7; // Activation energy in eV (typical for Cu)
        const T = maxTemp;
        const stressFactor = Math.max(0.01, 1 - (maxStress / 1e9));

        const mtf = 50 * stressFactor * Math.exp(Ea / (kB * T)) / Math.exp(Ea / (kB * 373));
        return Math.max(0.1, Math.min(100, mtf));
    }

    /**
     * Get data for visualization: returns object with typed arrays
     */
    getVisualizationData() {
        return {
            temperature: this.temperature,
            stress: this.stress,
            materialGrid: this.materialGrid,
            dimX: this.config.dimX,
            dimY: this.config.dimY,
            dimZ: this.config.dimZ
        };
    }
}
