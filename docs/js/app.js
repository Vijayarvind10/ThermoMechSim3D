/**
 * ThermoMechSim3D - Main Application Controller
 * Connects UI, simulation engine, and 3D visualization
 */

(function () {
    'use strict';

    // Application state
    let simulation = null;
    let visualizer = null;
    let currentPreset = 'hbm_stack';

    // DOM References
    const elements = {
        // Controls
        presetSelect: document.getElementById('preset-select'),
        dimX: document.getElementById('dim-x'),
        dimY: document.getElementById('dim-y'),
        dimZ: document.getElementById('dim-z'),
        dimXVal: document.getElementById('dim-x-val'),
        dimYVal: document.getElementById('dim-y-val'),
        dimZVal: document.getElementById('dim-z-val'),
        totalPower: document.getElementById('total-power'),
        ambientTemp: document.getElementById('ambient-temp'),
        refTemp: document.getElementById('ref-temp'),
        timeStep: document.getElementById('time-step'),
        maxSteps: document.getElementById('max-steps'),
        convergence: document.getElementById('convergence'),
        stressThreshold: document.getElementById('stress-threshold'),

        // Buttons
        btnRun: document.getElementById('btn-run'),
        btnStop: document.getElementById('btn-stop'),
        btnReset: document.getElementById('btn-reset'),

        // Progress
        progressSection: document.getElementById('progress-section'),
        progressFill: document.getElementById('progress-fill'),
        progressText: document.getElementById('progress-text'),
        progressDetail: document.getElementById('progress-detail'),

        // Status
        simStatus: document.getElementById('sim-status'),

        // Results
        resMaxTemp: document.getElementById('res-max-temp'),
        resMinTemp: document.getElementById('res-min-temp'),
        resMaxStress: document.getElementById('res-max-stress'),
        resCritical: document.getElementById('res-critical'),
        resConvergence: document.getElementById('res-convergence'),
        resStep: document.getElementById('res-step'),
        resElapsed: document.getElementById('res-elapsed'),

        // Info panels
        stackInfo: document.getElementById('stack-info'),
        materialsInfo: document.getElementById('materials-info'),
        failureSection: document.getElementById('failure-section'),
        failureInfo: document.getElementById('failure-info'),

        // Visualization controls
        sliceZ: document.getElementById('slice-z'),
        sliceZVal: document.getElementById('slice-z-val'),
        opacitySlider: document.getElementById('opacity'),
        showWireframe: document.getElementById('show-wireframe'),
        showCritical: document.getElementById('show-critical'),

        // Legend
        legendTitle: document.getElementById('legend-title'),
        legendMin: document.getElementById('legend-min'),
        legendMax: document.getElementById('legend-max'),

        // Canvas
        canvas: document.getElementById('render-canvas')
    };

    /**
     * Initialize the application
     */
    function init() {
        // Create simulation engine
        simulation = new ThermoMechSimulation();

        // Create visualizer
        visualizer = new SimulationVisualizer(elements.canvas);

        // Bind event listeners
        bindEvents();

        // Load initial preset
        loadPreset('hbm_stack');

        // Initialize visualization with default data
        initializeDefaultView();
    }

    /**
     * Bind all UI event listeners
     */
    function bindEvents() {
        // Preset selector
        elements.presetSelect.addEventListener('change', function () {
            loadPreset(this.value);
        });

        // Range sliders
        elements.dimX.addEventListener('input', function () {
            elements.dimXVal.textContent = this.value;
        });
        elements.dimY.addEventListener('input', function () {
            elements.dimYVal.textContent = this.value;
        });
        elements.dimZ.addEventListener('input', function () {
            elements.dimZVal.textContent = this.value;
        });

        // Buttons
        elements.btnRun.addEventListener('click', runSimulation);
        elements.btnStop.addEventListener('click', stopSimulation);
        elements.btnReset.addEventListener('click', resetSimulation);

        // Visualization mode buttons
        document.querySelectorAll('.viz-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                document.querySelectorAll('.viz-btn').forEach(function (b) {
                    b.classList.remove('active');
                });
                this.classList.add('active');
                const mode = this.getAttribute('data-mode');
                visualizer.setViewMode(mode);
                updateLegend(mode);
            });
        });

        // Slice control
        elements.sliceZ.addEventListener('input', function () {
            const val = parseInt(this.value);
            if (val >= 100) {
                elements.sliceZVal.textContent = 'All';
            } else {
                elements.sliceZVal.textContent = val + '%';
            }
            visualizer.setSliceZ(val / 100);
        });

        // Opacity control
        elements.opacitySlider.addEventListener('input', function () {
            visualizer.setOpacity(parseInt(this.value) / 100);
        });

        // Wireframe toggle
        elements.showWireframe.addEventListener('change', function () {
            visualizer.setWireframe(this.checked);
        });

        // Critical points toggle
        elements.showCritical.addEventListener('change', function () {
            visualizer.setShowCritical(this.checked);
        });
    }

    /**
     * Load a preset configuration
     */
    function loadPreset(presetKey) {
        currentPreset = presetKey;
        const preset = PRESETS[presetKey];
        if (!preset) return;

        // Update stack info panel
        updateStackInfo(preset);

        // Update materials info
        updateMaterialsInfo(preset);
    }

    /**
     * Initialize default visualization (ambient temperature everywhere)
     */
    function initializeDefaultView() {
        const config = getConfig();
        simulation.initialize(config, currentPreset);

        const visData = simulation.getVisualizationData();
        visualizer.updateData(visData);
        visualizer.setViewMode('material');

        // Set material view as default before simulation runs
        document.querySelectorAll('.viz-btn').forEach(function (b) {
            b.classList.remove('active');
        });
        document.querySelector('.viz-btn[data-mode="material"]').classList.add('active');
        updateLegend('material');
    }

    /**
     * Get simulation configuration from UI inputs
     */
    function getConfig() {
        const dimX = parseInt(elements.dimX.value);
        const dimY = parseInt(elements.dimY.value);
        const dimZ = parseInt(elements.dimZ.value);
        const gridSpacing = 1e-6; // 1 micron

        return {
            dimX: dimX,
            dimY: dimY,
            dimZ: dimZ,
            dx: gridSpacing,
            dy: gridSpacing,
            dz: gridSpacing,
            dt: parseFloat(elements.timeStep.value) * 1e-9,
            totalPower: parseFloat(elements.totalPower.value),
            ambientTemp: parseFloat(elements.ambientTemp.value),
            refTemp: parseFloat(elements.refTemp.value),
            maxSteps: parseInt(elements.maxSteps.value),
            convergenceThreshold: parseFloat(elements.convergence.value),
            stressThreshold: parseFloat(elements.stressThreshold.value) * 1e6
        };
    }

    /**
     * Run the simulation
     */
    async function runSimulation() {
        const config = getConfig();

        // Update UI state
        setRunning(true);

        // Initialize simulation
        simulation.initialize(config, currentPreset);

        // Set up callbacks
        simulation.onProgress = function (progress) {
            elements.progressFill.style.width = (progress * 100) + '%';
            elements.progressText.textContent = Math.round(progress * 100) + '%';
        };

        simulation.onStepUpdate = function (data) {
            elements.resMaxTemp.textContent = data.maxTemp.toFixed(2) + ' K';
            elements.resMinTemp.textContent = data.minTemp.toFixed(2) + ' K';
            elements.resMaxStress.textContent = (data.maxStress / 1e6).toFixed(2) + ' MPa';
            elements.resConvergence.textContent = data.convergence.toExponential(2);
            elements.resStep.textContent = data.step + ' / ' + data.maxSteps;
            elements.resElapsed.textContent = data.elapsed.toFixed(1) + 's';
            elements.progressDetail.textContent = 'T_max: ' + data.maxTemp.toFixed(1) + 'K | Stress_max: ' + (data.maxStress / 1e6).toFixed(1) + ' MPa';

            // Update 3D visualization periodically
            const visData = simulation.getVisualizationData();
            visualizer.updateData(visData);
            updateLegendValues();
        };

        simulation.onComplete = function (results) {
            setRunning(false);

            // Final results update
            elements.resMaxTemp.textContent = results.maxTemp.toFixed(2) + ' K';
            elements.resMinTemp.textContent = results.minTemp.toFixed(2) + ' K';
            elements.resMaxStress.textContent = (results.maxStress / 1e6).toFixed(2) + ' MPa';
            elements.resCritical.textContent = results.criticalRegions.toLocaleString();
            elements.resConvergence.textContent = results.convergence.toExponential(2);
            elements.resStep.textContent = results.step;
            elements.resElapsed.textContent = results.elapsed.toFixed(2) + 's';

            // Switch to temperature view
            document.querySelectorAll('.viz-btn').forEach(function (b) {
                b.classList.remove('active');
            });
            document.querySelector('.viz-btn[data-mode="temperature"]').classList.add('active');
            visualizer.setViewMode('temperature');
            updateLegend('temperature');

            // Final visualization update
            const visData = simulation.getVisualizationData();
            visualizer.updateData(visData);
            updateLegendValues();

            // Show failure risk assessment
            showFailureRisk(results.failureRisk);

            // Mark status as done
            elements.simStatus.textContent = 'Completed';
            elements.simStatus.className = 'status-badge status-done';
            elements.progressFill.style.width = '100%';
            elements.progressText.textContent = 'Complete';
        };

        // Run simulation
        try {
            await simulation.run();
        } catch (err) {
            console.error('Simulation error:', err);
            elements.simStatus.textContent = 'Error';
            elements.simStatus.className = 'status-badge status-error';
            setRunning(false);
        }
    }

    /**
     * Stop the simulation
     */
    function stopSimulation() {
        if (simulation) {
            simulation.stop();
        }
        setRunning(false);
        elements.simStatus.textContent = 'Stopped';
        elements.simStatus.className = 'status-badge status-idle';
    }

    /**
     * Reset the simulation
     */
    function resetSimulation() {
        if (simulation && simulation.running) {
            simulation.stop();
        }
        setRunning(false);

        // Clear results
        elements.resMaxTemp.textContent = '--';
        elements.resMinTemp.textContent = '--';
        elements.resMaxStress.textContent = '--';
        elements.resCritical.textContent = '--';
        elements.resConvergence.textContent = '--';
        elements.resStep.textContent = '--';
        elements.resElapsed.textContent = '--';

        // Reset progress
        elements.progressFill.style.width = '0%';
        elements.progressText.textContent = '0%';
        elements.progressDetail.textContent = '';

        // Hide failure section
        elements.failureSection.style.display = 'none';

        // Reset status
        elements.simStatus.textContent = 'Idle';
        elements.simStatus.className = 'status-badge status-idle';

        // Re-initialize default view
        initializeDefaultView();
    }

    /**
     * Update UI running state
     */
    function setRunning(running) {
        elements.btnRun.disabled = running;
        elements.btnStop.disabled = !running;
        elements.progressSection.style.display = running ? 'block' : (simulation && simulation.step > 0 ? 'block' : 'none');

        if (running) {
            elements.simStatus.textContent = 'Running';
            elements.simStatus.className = 'status-badge status-running';

            // Disable configuration inputs during simulation
            elements.presetSelect.disabled = true;
            elements.dimX.disabled = true;
            elements.dimY.disabled = true;
            elements.dimZ.disabled = true;
        } else {
            elements.presetSelect.disabled = false;
            elements.dimX.disabled = false;
            elements.dimY.disabled = false;
            elements.dimZ.disabled = false;
        }
    }

    /**
     * Update the color legend
     */
    function updateLegend(mode) {
        switch (mode) {
            case 'temperature':
                elements.legendTitle.textContent = 'Temperature (K)';
                break;
            case 'stress':
                elements.legendTitle.textContent = 'Stress (MPa)';
                break;
            case 'material':
                elements.legendTitle.textContent = 'Material';
                break;
        }
        updateLegendValues();
    }

    /**
     * Update legend min/max values from current data
     */
    function updateLegendValues() {
        const range = visualizer.getDataRange();
        if (!range) return;

        const mode = visualizer.viewMode;
        if (mode === 'temperature') {
            elements.legendMin.textContent = range.min.toFixed(1);
            elements.legendMax.textContent = range.max.toFixed(1);
        } else if (mode === 'stress') {
            elements.legendMin.textContent = (range.min / 1e6).toFixed(1);
            elements.legendMax.textContent = (range.max / 1e6).toFixed(1);
        } else {
            elements.legendMin.textContent = '';
            elements.legendMax.textContent = '';
        }
    }

    /**
     * Update the stack info panel
     */
    function updateStackInfo(preset) {
        let html = '';
        preset.stackup.forEach(function (layer) {
            const mat = MATERIALS[layer.material];
            const color = mat ? mat.color : '#888';
            const thickness = layer.thickness * 1e6; // Convert to microns
            html += '<div class="stack-layer">';
            html += '<div class="layer-color" style="background:' + color + '"></div>';
            html += '<span class="layer-name">' + layer.name + '</span>';
            html += '<span class="layer-detail">' + thickness.toFixed(0) + ' um</span>';
            html += '</div>';
        });
        elements.stackInfo.innerHTML = html;
    }

    /**
     * Update the materials info panel
     */
    function updateMaterialsInfo(preset) {
        // Collect unique materials in this preset
        const usedMaterials = new Set();
        preset.stackup.forEach(function (layer) {
            usedMaterials.add(layer.material);
        });

        let html = '';
        usedMaterials.forEach(function (matName) {
            const mat = MATERIALS[matName];
            if (!mat) return;
            html += '<div class="material-item">';
            html += '<span class="material-name">' + mat.name + '</span>';
            html += ' <span class="material-prop">(' + mat.category + ')</span>';
            html += '<br><span class="material-prop">';
            html += 'k=' + mat.thermal_conductivity + ' W/mK, ';
            html += 'E=' + (mat.youngs_modulus / 1e9).toFixed(0) + ' GPa, ';
            html += 'CTE=' + (mat.thermal_expansion * 1e6).toFixed(1) + ' ppm/K';
            html += '</span>';
            html += '</div>';
        });
        elements.materialsInfo.innerHTML = html;
    }

    /**
     * Show failure risk assessment results
     */
    function showFailureRisk(risks) {
        elements.failureSection.style.display = 'block';

        let html = '';
        risks.forEach(function (risk) {
            const statusClass = risk.status === 'ok' ? 'risk-ok' :
                risk.status === 'warn' ? 'risk-warn' : 'risk-critical';

            html += '<div class="risk-item">';
            html += '<div class="risk-label">' + risk.component + '</div>';
            if (risk.value) {
                html += '<div class="risk-value ' + statusClass + '">' + risk.value + '</div>';
            } else {
                html += '<div class="risk-value ' + statusClass + '">' + (risk.margin * 100).toFixed(1) + '% margin</div>';
            }
            html += '</div>';
        });

        elements.failureInfo.innerHTML = html;
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
