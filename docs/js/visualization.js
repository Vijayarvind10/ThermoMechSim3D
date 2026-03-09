/**
 * ThermoMechSim3D - 3D Visualization Engine using Three.js
 * Renders temperature, stress, and material fields as interactive 3D voxel views
 */

class SimulationVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.voxelGroup = null;
        this.criticalGroup = null;
        this.axesHelper = null;

        this.viewMode = 'temperature'; // 'temperature', 'stress', 'material'
        this.sliceZ = 1.0; // 0-1, fraction of Z to show
        this.opacity = 0.8;
        this.showWireframe = false;
        this.showCritical = true;

        this.data = null;
        this.instancedMesh = null;
        this.colorAttribute = null;
        this.dummy = new THREE.Object3D();

        this._init();
        this._animate();
    }

    _init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0d1117);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(3, 2, 3);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 20;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 8, 5);
        this.scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-3, -2, -5);
        this.scene.add(dirLight2);

        // Groups
        this.voxelGroup = new THREE.Group();
        this.scene.add(this.voxelGroup);

        this.criticalGroup = new THREE.Group();
        this.scene.add(this.criticalGroup);

        // Axes helper
        this.axesHelper = new THREE.AxesHelper(1.5);
        this.scene.add(this.axesHelper);

        // Grid helper
        const gridHelper = new THREE.GridHelper(4, 20, 0x30363d, 0x21262d);
        gridHelper.position.y = -0.5;
        this.scene.add(gridHelper);

        // Handle resize
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(this.canvas.parentElement);
    }

    _onResize() {
        const parent = this.canvas.parentElement;
        const width = parent.clientWidth;
        const height = parent.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Update visualization with new simulation data
     */
    updateData(simData) {
        this.data = simData;
        this._rebuildVisualization();
    }

    /**
     * Set the view mode
     */
    setViewMode(mode) {
        this.viewMode = mode;
        if (this.data) {
            this._updateColors();
        }
    }

    /**
     * Set Z slice (0-1)
     */
    setSliceZ(fraction) {
        this.sliceZ = fraction;
        if (this.data) {
            this._rebuildVisualization();
        }
    }

    /**
     * Set voxel opacity
     */
    setOpacity(value) {
        this.opacity = value;
        if (this.instancedMesh) {
            this.instancedMesh.material.opacity = value;
        }
    }

    /**
     * Toggle wireframe
     */
    setWireframe(enabled) {
        this.showWireframe = enabled;
        if (this.instancedMesh) {
            this.instancedMesh.material.wireframe = enabled;
        }
    }

    /**
     * Toggle critical points
     */
    setShowCritical(enabled) {
        this.showCritical = enabled;
        if (this.criticalGroup) {
            this.criticalGroup.visible = enabled;
        }
    }

    /**
     * Rebuild the entire 3D visualization
     */
    _rebuildVisualization() {
        if (!this.data) return;

        const { dimX, dimY, dimZ, temperature, stress, materialGrid } = this.data;

        // Clear previous meshes
        while (this.voxelGroup.children.length > 0) {
            const child = this.voxelGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.voxelGroup.remove(child);
        }

        while (this.criticalGroup.children.length > 0) {
            const child = this.criticalGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.criticalGroup.remove(child);
        }

        // Calculate visible Z layers
        const maxZ = Math.max(1, Math.ceil(this.sliceZ * dimZ));

        // Determine sampling rate to keep voxel count manageable
        // Max ~50000 visible voxels for performance
        const totalVisible = dimX * dimY * maxZ;
        let step = 1;
        if (totalVisible > 50000) {
            step = Math.ceil(Math.cbrt(totalVisible / 50000));
        }

        // Count visible voxels
        let visibleCount = 0;
        for (let z = 0; z < maxZ; z += step) {
            for (let y = 0; y < dimY; y += step) {
                for (let x = 0; x < dimX; x += step) {
                    visibleCount++;
                }
            }
        }

        if (visibleCount === 0) return;

        // Create instanced mesh
        const voxelSize = 2.0 / Math.max(dimX, dimY, dimZ);
        const geometry = new THREE.BoxGeometry(voxelSize * 0.9, voxelSize * 0.9, voxelSize * 0.9);
        const material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: this.opacity,
            wireframe: this.showWireframe,
            side: THREE.FrontSide,
            shininess: 30
        });

        this.instancedMesh = new THREE.InstancedMesh(geometry, material, visibleCount);
        this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        // Create color buffer
        const colors = new Float32Array(visibleCount * 3);

        // Position voxels and set colors
        let instanceIdx = 0;
        const offsetX = -(dimX * voxelSize) / 2;
        const offsetY = -(dimY * voxelSize) / 2;
        const offsetZ = -(maxZ * voxelSize) / 2;

        // Compute min/max for color mapping
        let minVal = Infinity, maxVal = -Infinity;
        const fieldData = this.viewMode === 'temperature' ? temperature :
                          this.viewMode === 'stress' ? stress : null;

        if (fieldData) {
            for (let i = 0; i < fieldData.length; i++) {
                if (fieldData[i] < minVal) minVal = fieldData[i];
                if (fieldData[i] > maxVal) maxVal = fieldData[i];
            }
        }

        // Store range for legend
        this._dataRange = { min: minVal, max: maxVal };

        for (let z = 0; z < maxZ; z += step) {
            for (let y = 0; y < dimY; y += step) {
                for (let x = 0; x < dimX; x += step) {
                    const dataIdx = z * dimX * dimY + y * dimX + x;

                    // Position
                    this.dummy.position.set(
                        offsetX + x * voxelSize,
                        offsetZ + z * voxelSize,  // Z maps to Y in 3D view (height)
                        offsetY + y * voxelSize
                    );
                    this.dummy.updateMatrix();
                    this.instancedMesh.setMatrixAt(instanceIdx, this.dummy.matrix);

                    // Color
                    let color;
                    if (this.viewMode === 'material') {
                        const matId = materialGrid[dataIdx];
                        const mat = getMaterialByIndex(matId);
                        color = new THREE.Color(mat.color);
                    } else {
                        const value = fieldData[dataIdx];
                        const t = maxVal > minVal ? (value - minVal) / (maxVal - minVal) : 0;
                        color = this._heatmapColor(t);
                    }

                    colors[instanceIdx * 3] = color.r;
                    colors[instanceIdx * 3 + 1] = color.g;
                    colors[instanceIdx * 3 + 2] = color.b;

                    instanceIdx++;
                }
            }
        }

        // Set instance colors
        this.instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
        this.instancedMesh.instanceMatrix.needsUpdate = true;

        this.voxelGroup.add(this.instancedMesh);

        // Add critical point markers if stress mode
        if (this.showCritical && this.viewMode === 'stress') {
            this._addCriticalMarkers(dimX, dimY, dimZ, maxZ, voxelSize, offsetX, offsetY, offsetZ, step);
        }
    }

    /**
     * Update only the colors (faster than full rebuild)
     */
    _updateColors() {
        if (!this.instancedMesh || !this.data) {
            this._rebuildVisualization();
            return;
        }

        // For simplicity in mode switching, rebuild
        this._rebuildVisualization();
    }

    /**
     * Add markers for critical stress regions
     */
    _addCriticalMarkers(dimX, dimY, dimZ, maxZ, voxelSize, offsetX, offsetY, offsetZ, step) {
        if (!this.data) return;

        const { stress } = this.data;
        const threshold = 100e6; // 100 MPa default
        const markerGeo = new THREE.SphereGeometry(voxelSize * 0.8, 8, 8);
        const markerMat = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.6
        });

        let count = 0;
        for (let z = 0; z < maxZ; z += step * 2) {
            for (let y = 0; y < dimY; y += step * 2) {
                for (let x = 0; x < dimX; x += step * 2) {
                    const idx = z * dimX * dimY + y * dimX + x;
                    if (stress[idx] > threshold && count < 100) {
                        const marker = new THREE.Mesh(markerGeo, markerMat);
                        marker.position.set(
                            offsetX + x * voxelSize,
                            offsetZ + z * voxelSize,
                            offsetY + y * voxelSize
                        );
                        this.criticalGroup.add(marker);
                        count++;
                    }
                }
            }
        }

        this.criticalGroup.visible = this.showCritical;
    }

    /**
     * Generate heatmap color from value 0-1
     * Blue -> Cyan -> Green -> Yellow -> Red
     */
    _heatmapColor(t) {
        t = Math.max(0, Math.min(1, t));
        let r, g, b;

        if (t < 0.25) {
            const s = t / 0.25;
            r = 0; g = s; b = 1;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0; g = 1; b = 1 - s;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = s; g = 1; b = 0;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1; g = 1 - s; b = 0;
        }

        return new THREE.Color(r, g, b);
    }

    /**
     * Get current data range for legend
     */
    getDataRange() {
        return this._dataRange || { min: 0, max: 1 };
    }

    /**
     * Reset camera to default position
     */
    resetCamera() {
        this.camera.position.set(3, 2, 3);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }

    /**
     * Dispose of all resources
     */
    dispose() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}
