/**
 * ThermoMechSim3D - Material Properties Database
 * Ported from C++ MaterialDatabase class
 */

const MATERIALS = {
    Si: {
        name: 'Si',
        category: 'semiconductor',
        thermal_conductivity: 149.0,   // W/(m*K)
        specific_heat: 700.0,          // J/(kg*K)
        density: 2329.0,               // kg/m^3
        youngs_modulus: 170.0e9,       // Pa
        poissons_ratio: 0.28,
        thermal_expansion: 2.6e-6,     // 1/K
        yield_strength: 7000.0e6,      // Pa
        ultimate_strength: 7000.0e6,   // Pa
        melting_point: 1687.0,         // K
        fracture_toughness: 0.9e6,     // Pa*m^(1/2)
        color: '#4a90d9'
    },
    Cu: {
        name: 'Cu',
        category: 'metal',
        thermal_conductivity: 400.0,
        specific_heat: 385.0,
        density: 8960.0,
        youngs_modulus: 130.0e9,
        poissons_ratio: 0.34,
        thermal_expansion: 16.5e-6,
        yield_strength: 70.0e6,
        ultimate_strength: 220.0e6,
        melting_point: 1358.0,
        fracture_toughness: 30.0e6,
        color: '#d4760a'
    },
    Cu_TSV: {
        name: 'Cu_TSV',
        category: 'metal',
        thermal_conductivity: 400.0,
        specific_heat: 385.0,
        density: 8960.0,
        youngs_modulus: 110.0e9,
        poissons_ratio: 0.34,
        thermal_expansion: 16.5e-6,
        yield_strength: 50.0e6,
        ultimate_strength: 220.0e6,
        melting_point: 1358.0,
        fracture_toughness: 30.0e6,
        color: '#c76b08'
    },
    SiO2: {
        name: 'SiO2',
        category: 'dielectric',
        thermal_conductivity: 1.4,
        specific_heat: 730.0,
        density: 2200.0,
        youngs_modulus: 70.0e9,
        poissons_ratio: 0.17,
        thermal_expansion: 0.5e-6,
        yield_strength: 8400.0e6,
        ultimate_strength: 8400.0e6,
        melting_point: 1986.0,
        fracture_toughness: 0.77e6,
        color: '#7eb8c9'
    },
    Al: {
        name: 'Al',
        category: 'metal',
        thermal_conductivity: 237.0,
        specific_heat: 897.0,
        density: 2700.0,
        youngs_modulus: 70.0e9,
        poissons_ratio: 0.35,
        thermal_expansion: 23.1e-6,
        yield_strength: 35.0e6,
        ultimate_strength: 90.0e6,
        melting_point: 933.0,
        fracture_toughness: 24.0e6,
        color: '#c0c0c0'
    },
    Underfill: {
        name: 'Underfill',
        category: 'polymer',
        thermal_conductivity: 0.3,
        specific_heat: 1100.0,
        density: 1200.0,
        youngs_modulus: 8.5e9,
        poissons_ratio: 0.35,
        thermal_expansion: 30.0e-6,
        yield_strength: 50.0e6,
        ultimate_strength: 70.0e6,
        melting_point: 473.0,
        fracture_toughness: 0.5e6,
        color: '#8b6914'
    },
    TIM: {
        name: 'TIM',
        category: 'thermal_interface',
        thermal_conductivity: 5.0,
        specific_heat: 1000.0,
        density: 2500.0,
        youngs_modulus: 5.0e9,
        poissons_ratio: 0.35,
        thermal_expansion: 25.0e-6,
        yield_strength: 5.0e6,
        ultimate_strength: 7.0e6,
        melting_point: 423.0,
        fracture_toughness: 0.3e6,
        color: '#a0a0a0'
    },
    SAC305: {
        name: 'SAC305',
        category: 'metal',
        thermal_conductivity: 58.0,
        specific_heat: 232.0,
        density: 7400.0,
        youngs_modulus: 51.0e9,
        poissons_ratio: 0.36,
        thermal_expansion: 21.0e-6,
        yield_strength: 32.0e6,
        ultimate_strength: 48.0e6,
        melting_point: 490.0,
        fracture_toughness: 1.8e6,
        color: '#808080'
    }
};

/**
 * Preset stack configurations
 */
const PRESETS = {
    hbm_stack: {
        name: 'HBM Memory Stack',
        description: 'High Bandwidth Memory with 4 DRAM dies + logic die',
        stackup: [
            { material: 'Si', thickness: 100e-6, name: 'Logic Die', hasPower: true, powerFraction: 0.5 },
            { material: 'Cu_TSV', thickness: 50e-6, name: 'TSV Layer 1', hasPower: false },
            { material: 'Si', thickness: 50e-6, name: 'DRAM Die 1', hasPower: true, powerFraction: 0.125 },
            { material: 'Cu_TSV', thickness: 50e-6, name: 'TSV Layer 2', hasPower: false },
            { material: 'Si', thickness: 50e-6, name: 'DRAM Die 2', hasPower: true, powerFraction: 0.125 },
            { material: 'Cu_TSV', thickness: 50e-6, name: 'TSV Layer 3', hasPower: false },
            { material: 'Si', thickness: 50e-6, name: 'DRAM Die 3', hasPower: true, powerFraction: 0.125 },
            { material: 'Cu_TSV', thickness: 50e-6, name: 'TSV Layer 4', hasPower: false },
            { material: 'Si', thickness: 50e-6, name: 'DRAM Die 4', hasPower: true, powerFraction: 0.125 },
            { material: 'SiO2', thickness: 20e-6, name: 'Interposer', hasPower: false }
        ],
        boundary: { top: 'convection', bottom: 'constant_temp', sides: 'adiabatic' },
        convectionCoeff: 1e4,
        bottomTemp: 343.15
    },
    logic_die: {
        name: 'Logic Die (Single)',
        description: 'Single logic die with heat sink',
        stackup: [
            { material: 'Cu', thickness: 200e-6, name: 'Heat Sink', hasPower: false },
            { material: 'TIM', thickness: 50e-6, name: 'TIM Layer', hasPower: false },
            { material: 'Si', thickness: 300e-6, name: 'Logic Die', hasPower: true, powerFraction: 1.0 },
            { material: 'SiO2', thickness: 50e-6, name: 'Oxide Layer', hasPower: false }
        ],
        boundary: { top: 'convection', bottom: 'constant_temp', sides: 'adiabatic' },
        convectionCoeff: 5e3,
        bottomTemp: 323.15
    },
    chiplet: {
        name: '2.5D Chiplet Package',
        description: 'Two compute chiplets on silicon interposer',
        stackup: [
            { material: 'Si', thickness: 150e-6, name: 'Chiplet A', hasPower: true, powerFraction: 0.45 },
            { material: 'SAC305', thickness: 30e-6, name: 'Microbumps', hasPower: false },
            { material: 'Si', thickness: 150e-6, name: 'Chiplet B', hasPower: true, powerFraction: 0.45 },
            { material: 'SAC305', thickness: 30e-6, name: 'C4 Bumps', hasPower: false },
            { material: 'SiO2', thickness: 100e-6, name: 'Si Interposer', hasPower: false },
            { material: 'Underfill', thickness: 50e-6, name: 'Underfill', hasPower: false },
            { material: 'Cu', thickness: 200e-6, name: 'Substrate', hasPower: false }
        ],
        boundary: { top: 'convection', bottom: 'constant_temp', sides: 'adiabatic' },
        convectionCoeff: 8e3,
        bottomTemp: 333.15
    },
    custom: {
        name: 'Custom',
        description: 'Define your own stack',
        stackup: [
            { material: 'Si', thickness: 200e-6, name: 'Die', hasPower: true, powerFraction: 1.0 }
        ],
        boundary: { top: 'convection', bottom: 'constant_temp', sides: 'adiabatic' },
        convectionCoeff: 5e3,
        bottomTemp: 323.15
    }
};

/**
 * Get material index from name for use in flat arrays
 */
function getMaterialIndex(materialName) {
    const names = Object.keys(MATERIALS);
    return names.indexOf(materialName);
}

/**
 * Get material by index
 */
function getMaterialByIndex(index) {
    const names = Object.keys(MATERIALS);
    if (index >= 0 && index < names.length) {
        return MATERIALS[names[index]];
    }
    return MATERIALS.Si; // Default fallback
}
