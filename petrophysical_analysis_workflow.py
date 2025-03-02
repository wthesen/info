from math import *
from TechlogMath import *
from operator import *
import sys
if sys.version_info[0]==3:
    from six.moves import range

PI     = 3.14159265358979323846
PIO2   = 1.57079632679489661923
PIO4   = 7.85398163397448309616E-1
SQRT2  = 1.41421356237309504880
SQRTH  = 7.07106781186547524401E-1
E      = exp(1)
LN2    = log(2)
LN10   = log(10)
LOG2E  = 1.4426950408889634073599
LOG10E = 1.0 / LN10
MissingValue = -9999
def iif(condition, trueResult=MissingValue, falseResult=MissingValue):
	if condition:
		return trueResult
	else:
		return falseResult

#Declarations
#The dictionary of parameters v2.0
#name,label,bname,type,family,measurement,unit,value,mode,description,group,min,max,list,enable,iscombocheckbox,isused
parameterDict = {}
try:
	if Parameter:
		pass
except NameError:
	class Parameter:
		def __init__(self, **d):
			pass

__author__ = """W T (wadet)"""
__date__ = """2025-03-01"""
__version__ = """1.0"""
__group__ = """"""
__suffix__ = """"""
__prefix__ = """"""
__applyMode__ = """0"""
__awiEngine__ = """v2"""
__layoutTemplateMode__ = """"""
__includeMissingValues__ = """True"""
__keepPreviouslyComputedValues__ = """True"""
__areInputDisplayed__ = """True"""
__useMultiWellLayout__ = """True"""
__useFamilyAssignmentRules__ = """True"""
__idForHelp__ = """"""
__executionGranularity__ = """full"""
#DeclarationsEnd
"""
Comprehensive Petrophysical Analysis Workflow

A complete workflow for organizing, analyzing, and evaluating well data
for pay mapping, lithology computation, and bypass pay identification.

Author: GitHub Copilot for wthesen
Created: 2025-03-02
"""

import os
import pandas as pd
import numpy as np
import glob
import time
import importlib  # Add this import
from datetime import datetime

class PetrophysicalWorkflow:
    """Main workflow class for petrophysical analysis"""
    
    def __init__(self, base_dir=None):
        """Initialize the workflow with a base directory"""
        if base_dir is None:
            self.base_dir = self._setup_project_directory()
        else:
            self.base_dir = base_dir
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {}
        for subdir in ['data', 'logs', 'maps', 'results', 'standardized_las', 
                      'charts', 'lithology', 'pay_analysis', 'summary']:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            self.dirs[subdir] = dir_path
        
        print(f"Project initialized at: {self.base_dir}")
        
    def _setup_project_directory(self):
        """Set up project directory with appropriate permissions"""
        # Try several potential locations with write permissions
        potential_locations = [
            os.path.join(os.path.expanduser("~"), "techlog_analysis"),  # User home
            os.path.join(os.environ.get("TEMP", "C:\\Temp"), "techlog_analysis"),  # Temp directory
            os.path.join("C:\\", "Temp", "techlog_analysis"),  # C:\Temp
            "D:\\techlog_analysis",  # D: drive if available
        ]
        
        for location in potential_locations:
            try:
                os.makedirs(location, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(location, f"test_write_permission_{int(time.time())}.txt")
                with open(test_file, 'w') as f:
                    f.write(f"Testing write permissions at {datetime.now()}")
                os.remove(test_file)
                return location
            except Exception as e:
                print(f"Cannot use {location}: {e}")
        
        raise PermissionError("Could not find a writable directory for the project")
    
    def create_master_well_index(self):
        """Create a master index of wells based on TLVARIABLE file data"""
        
        print("Creating master well index...")
        
        # Well names from the TLVARIABLE file
        well_names = [
            "100_01-04-081-24W4_00", "100_01-09-082-25W4_00", "100_01-09-082-25W4_001", 
            "100_01-09-082-25W4_002", "100_01-26-081-25W4_00", "100_01-26-081-25W4_001", 
            "100_01-28-081-25W4_00", "100_01-28-081-25W4_001", "100_01-28-081-25W4_002",
            "100_02-36-081-25W4_00", "100_02-36-081-25W4_001", "100_03-09-081-25W4_00",
            "100_03-09-081-25W4_001", "100_03-09-081-25W4_002", "100_03-27-081-25W4_00",
            "100_03-27-081-25W4_001", "100_04-04-081-24W4_00", "100_04-17-081-24W4_00"
        ]
        
        # Create initial dataframe
        wells_df = pd.DataFrame({
            'well_name': well_names,
            'uwi': None,
            'location': None,
            'kb_elevation': None,
            'ground_elevation': None,
            'total_depth': None,
            'creation_date': datetime.now().strftime("%Y-%m-%d"),
            'has_logs': False,
            'data_quality': None
        })
        
        # Extract location information
        wells_df['location'] = wells_df['well_name'].apply(
            lambda x: x.split('_')[1] if '_' in x and len(x.split('_')) > 1 else None
        )
        
        # Add placeholder columns for key log availability
        key_logs = ['GR', 'RHOB', 'NPHI', 'RT', 'DT', 'PE', 'CALI']
        for log in key_logs:
            wells_df[f'has_{log}'] = False
        
        # Save the master index
        output_file = os.path.join(self.base_dir, "master_well_index.csv")
        wells_df.to_csv(output_file, index=False)
        print(f"Created master well index with {len(wells_df)} wells")
        
        return wells_df
    
    def create_log_inventory(self):
        """Create a log curve inventory based on TLVARIABLE file data"""
        
        print("Creating log curve inventory...")
        
        # Sample data based on TLVARIABLE file information
        curve_data = [
            {'well_name': '100_01-04-081-24W4_00', 'curve_name': 'VSH', 'family': 'Shale Volume Fraction', 'unit': 'v/v'},
            {'well_name': '100_01-04-081-24W4_00', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-09-082-25W4_00', 'curve_name': 'RHOB', 'family': 'Bulk Density', 'unit': 'G/C3'},
            {'well_name': '100_01-09-082-25W4_00', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-09-082-25W4_001', 'curve_name': 'DEPTH', 'family': 'Measured Depth', 'unit': 'M'},
            {'well_name': '100_01-09-082-25W4_001', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-09-082-25W4_002', 'curve_name': 'DEPTH', 'family': 'Measured Depth', 'unit': 'M'},
            {'well_name': '100_01-09-082-25W4_002', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-26-081-25W4_00', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'GAPI'},
            {'well_name': '100_01-26-081-25W4_00', 'curve_name': 'RHOB', 'family': 'Bulk Density', 'unit': 'G/C3'},
            {'well_name': '100_01-26-081-25W4_001', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'GAPI'},
            {'well_name': '100_01-28-081-25W4_00', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-28-081-25W4_00', 'curve_name': 'NPHI', 'family': 'Neutron Porosity', 'unit': 'V/V'},
            {'well_name': '100_01-28-081-25W4_001', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_01-28-081-25W4_002', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'API'},
            {'well_name': '100_02-36-081-25W4_00', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'GAPI'},
            {'well_name': '100_02-36-081-25W4_001', 'curve_name': 'GR', 'family': 'Gamma Ray', 'unit': 'GAPI'}
        ]
        
        # Add sample min/max values
        for curve in curve_data:
            if curve['curve_name'] == 'GR':
                curve['min_value'] = 20.0
                curve['max_value'] = 180.0
            elif curve['curve_name'] == 'RHOB':
                curve['min_value'] = 1.96
                curve['max_value'] = 2.66
            elif curve['curve_name'] == 'NPHI':
                curve['min_value'] = 0.0
                curve['max_value'] = 0.45
            elif curve['curve_name'] == 'DEPTH':
                curve['min_value'] = 0.0
                curve['max_value'] = 500.0
            else:
                curve['min_value'] = None
                curve['max_value'] = None
            
            curve['quality'] = 'Good'
            curve['data_coverage'] = 0.95  # Example coverage value
        
        # Create DataFrame
        curves_df = pd.DataFrame(curve_data)
        
        # Save the log inventory
        output_file = os.path.join(self.base_dir, "log_curve_inventory.csv")
        curves_df.to_csv(output_file, index=False)
        print(f"Created log curve inventory with {len(curves_df)} entries")
        
        return curves_df
    
    def assess_data_quality(self):
        """Assess data quality based on well and curve information"""
        
        print("Assessing data quality...")
        
        # Load master well index and curve inventory
        try:
            wells_file = os.path.join(self.base_dir, "master_well_index.csv")
            curves_file = os.path.join(self.base_dir, "log_curve_inventory.csv")
            
            if not os.path.exists(wells_file) or not os.path.exists(curves_file):
                print("Creating required files first...")
                self.create_master_well_index()
                self.create_log_inventory()
                
                wells_df = pd.read_csv(wells_file)
                curves_df = pd.read_csv(curves_file)
            else:
                wells_df = pd.read_csv(wells_file)
                curves_df = pd.read_csv(curves_file)
            
            print(f"Loaded {len(wells_df)} wells and {len(curves_df)} curve entries")
        except Exception as e:
            print(f"Error loading required files: {str(e)}")
            return None
        
        # Assess quality for each well
        quality_results = []
        
        for _, well in wells_df.iterrows():
            well_name = well['well_name']
            
            # Get curves for this well
            well_curves = curves_df[curves_df['well_name'] == well_name]
            
            # Check for key log types
            has_gr = any(well_curves['curve_name'].str.upper() == 'GR')
            has_density = any(well_curves['curve_name'].str.upper() == 'RHOB')
            has_neutron = any(well_curves['curve_name'].str.upper() == 'NPHI')
            has_resistivity = any(well_curves['curve_name'].str.upper().isin(['RT', 'RESD', 'ILD']))
            
            # Update has_X flags in wells_df
            idx = wells_df[wells_df['well_name'] == well_name].index
            if len(idx) > 0:
                wells_df.loc[idx[0], 'has_GR'] = has_gr
                wells_df.loc[idx[0], 'has_RHOB'] = has_density
                wells_df.loc[idx[0], 'has_NPHI'] = has_neutron
                wells_df.loc[idx[0], 'has_RT'] = has_resistivity
            
            # Determine overall quality
            if has_gr and has_density and has_neutron and has_resistivity:
                quality = 'Good'
            elif has_gr and (has_density or has_neutron or has_resistivity):
                quality = 'Fair'
            elif has_gr or has_density or has_neutron or has_resistivity:
                quality = 'Poor'
            else:
                quality = 'Unusable'
            
            # Update quality in wells_df
            if len(idx) > 0:
                wells_df.loc[idx[0], 'data_quality'] = quality
            
            # Record quality assessment
            quality_results.append({
                'well_name': well_name,
                'has_gr': has_gr,
                'has_density': has_density,
                'has_neutron': has_neutron,
                'has_resistivity': has_resistivity,
                'curve_count': len(well_curves),
                'quality': quality
            })
        
        # Create quality dataframe
        quality_df = pd.DataFrame(quality_results)
        
        # Save results
        try:
            quality_file = os.path.join(self.base_dir, "quality_assessment.csv")
            quality_df.to_csv(quality_file, index=False)
            
            # Save updated well index
            wells_df.to_csv(wells_file, index=False)
            
            print(f"Successfully assessed quality for {len(quality_df)} wells")
            print(f"Quality assessment saved to: {quality_file}")
            
            # Print quality summary
            print("\nQuality Summary:")
            print(quality_df['quality'].value_counts())
        except Exception as e:
            print(f"Error saving quality assessment: {str(e)}")
        
        return quality_df
    
    def prepare_lithology_data(self):
        """Create template data for lithology computation"""
        
        print("Preparing data for lithology computation...")
        
        # Load quality assessment to find good wells
        try:
            quality_file = os.path.join(self.base_dir, "quality_assessment.csv")
            if os.path.exists(quality_file):
                quality_df = pd.read_csv(quality_file)
                good_wells = quality_df[quality_df['quality'] == 'Good']['well_name'].tolist()
            else:
                # Use a subset of wells if quality assessment isn't available
                good_wells = [
                    "100_01-09-082-25W4_00", 
                    "100_01-26-081-25W4_00", 
                    "100_01-28-081-25W4_00"
                ]
        except Exception as e:
            print(f"Error loading quality assessment: {str(e)}")
            good_wells = ["100_01-09-082-25W4_00"]
        
        # Create sample data for lithology computation
        print(f"Creating lithology templates for {len(good_wells)} wells")
        
              # Define lithology types and their properties
        lithologies = {
            'Sandstone': {'GR_mean': 65, 'RHOB_mean': 2.35, 'NPHI_mean': 0.15, 'RT_mean': 20},
            'Shale': {'GR_mean': 120, 'RHOB_mean': 2.55, 'NPHI_mean': 0.28, 'RT_mean': 5},
            'Limestone': {'GR_mean': 40, 'RHOB_mean': 2.71, 'NPHI_mean': 0.08, 'RT_mean': 50},
            'Dolomite': {'GR_mean': 30, 'RHOB_mean': 2.85, 'NPHI_mean': 0.04, 'RT_mean': 100},
            'Coal': {'GR_mean': 35, 'RHOB_mean': 1.8, 'NPHI_mean': 0.35, 'RT_mean': 200}
        }
        
        # Create matrix property table
        matrix_df = pd.DataFrame([
            {'mineral': 'Quartz', 'density': 2.65, 'neutron': -0.03, 'pe': 1.8},
            {'mineral': 'Calcite', 'density': 2.71, 'neutron': 0.0, 'pe': 5.1},
            {'mineral': 'Dolomite', 'density': 2.87, 'neutron': 0.02, 'pe': 3.1},
            {'mineral': 'Illite', 'density': 2.53, 'neutron': 0.30, 'pe': 3.5},
            {'mineral': 'Kaolinite', 'density': 2.42, 'neutron': 0.37, 'pe': 1.8},
            {'mineral': 'Anhydrite', 'density': 2.98, 'neutron': 0.0, 'pe': 5.0},
            {'mineral': 'Coal', 'density': 1.8, 'neutron': 0.7, 'pe': 0.2}
        ])
        
        # Save matrix properties
        matrix_file = os.path.join(self.dirs['lithology'], "matrix_properties.csv")
        matrix_df.to_csv(matrix_file, index=False)
        print(f"Created matrix properties file: {matrix_file}")
        
        # Create sample lithology templates for each well
        for well in good_wells:
            # Create a sample dataset with depth intervals
            depths = np.arange(100, 500, 0.1)
            np.random.seed(42)  # For reproducibility
            
            # Create dataframe with curve data
            well_data = pd.DataFrame({
                'DEPTH': depths,
                'GR': np.random.normal(70, 30, len(depths)),
                'RHOB': np.random.normal(2.5, 0.2, len(depths)),
                'NPHI': np.random.normal(0.18, 0.08, len(depths)),
                'RT': np.random.lognormal(1, 1, len(depths))
            })
            
            # Create zones with different lithologies
            n_depths = len(depths)
            zone_size = n_depths // 5  # 5 zones
            
            # Assign lithologies to zones
            litho_types = list(lithologies.keys())
            for i, litho in enumerate(litho_types):
                if i >= 5:  # Only using 5 zones
                    break
                    
                start_idx = i * zone_size
                end_idx = (i + 1) * zone_size if i < 4 else n_depths
                
                props = lithologies[litho]
                
                # Adjust curves in this zone toward this lithology
                well_data.loc[start_idx:end_idx, 'GR'] = np.random.normal(
                    props['GR_mean'], 15, end_idx - start_idx)
                well_data.loc[start_idx:end_idx, 'RHOB'] = np.random.normal(
                    props['RHOB_mean'], 0.1, end_idx - start_idx)
                well_data.loc[start_idx:end_idx, 'NPHI'] = np.random.normal(
                    props['NPHI_mean'], 0.04, end_idx - start_idx)
                well_data.loc[start_idx:end_idx, 'RT'] = np.random.lognormal(
                    np.log(props['RT_mean']), 0.5, end_idx - start_idx)
            
            # Add computed VSH column
            gr_min = 30
            gr_max = 120
            well_data['VSH'] = (well_data['GR'] - gr_min) / (gr_max - gr_min)
            well_data['VSH'] = well_data['VSH'].clip(0, 1)  # Constrain to [0,1]
            
            # Add computed PHIE column (effective porosity)
            well_data['PHIE'] = (0.3 - 0.3 * well_data['VSH']) * (1.0 - well_data['RHOB'] / 2.65)
            well_data['PHIE'] = well_data['PHIE'].clip(0, 0.4)  # Constrain to reasonable values
            
            # Save well data
            output_file = os.path.join(self.dirs['lithology'], f"{well}_litho_data.csv")
            well_data.to_csv(output_file, index=False)
            print(f"Created lithology data for {well}")
        
        # Create lithology interpretation template
        litho_template_data = [{
            'well_name': good_wells[0] if good_wells else '100_01-09-082-25W4_00',
            'top_depth': 150.0,
            'bottom_depth': 155.0,
            'thickness': 5.0,
            'formation': 'McMurray',
            'lithology_primary': 'Sandstone',
            'lithology_secondary': 'Shale',
            'vsh_avg': 0.15,
            'phie_avg': 0.18,
            'sw_avg': 0.35,
            'pay_flag': 'Pay',
            'perf_flag': 'Not Perforated',
            'notes': 'Potential bypass pay'
        }]
        
        litho_template = pd.DataFrame(litho_template_data)
        
        # Save the template
        template_file = os.path.join(self.dirs['lithology'], "lithology_interpretation_template.csv")
        litho_template.to_csv(template_file, index=False)
        
        print(f"Created lithology interpretation template at {template_file}")
        return self.dirs['lithology']
    
    def prepare_pay_analysis(self):
        """Prepare data and templates for pay zone analysis"""
        
        print("Preparing data for pay zone analysis...")
        
        # Create pay cutoff criteria file
        cutoffs = pd.DataFrame([
            {'formation': 'McMurray', 'vsh_cutoff': 0.4, 'phie_cutoff': 0.08, 'sw_cutoff': 0.6, 'k_cutoff': 10.0},
            {'formation': 'Clearwater', 'vsh_cutoff': 0.5, 'phie_cutoff': 0.06, 'sw_cutoff': 0.65, 'k_cutoff': 1.0},
            {'formation': 'Grand Rapids', 'vsh_cutoff': 0.45, 'phie_cutoff': 0.07, 'sw_cutoff': 0.6, 'k_cutoff': 5.0}
        ])
        
        cutoffs_file = os.path.join(self.dirs['pay_analysis'], "pay_cutoffs.csv")
        cutoffs.to_csv(cutoffs_file, index=False)
        print(f"Created pay cutoff criteria file: {cutoffs_file}")
        
        # Create pay flag templates
        pay_template_data = [
            {
                'well_name': '100_01-09-082-25W4_00',
                'md_top': 150.0,
                'md_bottom': 160.0,
                'thickness': 10.0,
                'formation': 'McMurray',
                'vsh': 0.20,
                'phie': 0.15,
                'sw': 0.40,
                'k_md': 100.0,
                'pay_flag': 'Pay',
                'net_to_gross': 0.8,
                'perf_status': 'Not Perforated',
                'bypass_candidate': 'Yes',
                'notes': 'Good quality reservoir, unperforated'
            },
            {
                'well_name': '100_01-09-082-25W4_00',
                'md_top': 180.0,
                'md_bottom': 190.0,
                'thickness': 10.0,
                'formation': 'McMurray',
                'vsh': 0.35,
                'phie': 0.10,
                'sw': 0.50,
                'k_md': 50.0,
                'pay_flag': 'Marginal Pay',
                'net_to_gross': 0.6,
                'perf_status': 'Not Perforated',
                'bypass_candidate': 'Maybe',
                'notes': 'Moderate quality reservoir, requires further evaluation'
            }
        ]
        
        pay_template = pd.DataFrame(pay_template_data)
        pay_template_file = os.path.join(self.dirs['pay_analysis'], "pay_analysis_template.csv")
        pay_template.to_csv(pay_template_file, index=False)
        print(f"Created pay analysis template: {pay_template_file}")
        
        # Create recompletion candidates template
        recomp_template_data = [
            {
                'well_name': '100_01-28-081-25W4_00',
                'md_top': 200.0,
                'md_bottom': 210.0,
                'thickness': 10.0,
                'formation': 'McMurray',
                'phie': 0.14,
                'sw': 0.45,
                'k_md': 80.0,
                'estimated_recovery': 10000,
                'priority': 'High',
                'mechanical_issues': 'None',
                'economics': 'Good',
                'notes': 'Strong bypass pay candidate'
            }
        ]
        
        recomp_template = pd.DataFrame(recomp_template_data)
        recomp_file = os.path.join(self.dirs['pay_analysis'], "recompletion_candidates.csv")
        recomp_template.to_csv(recomp_file, index=False)
        print(f"Created recompletion candidates template: {recomp_file}")
        
        # Create a simple mapping template with sample data
        map_data = []
        
        # Generate some sample points based on well names from TLVARIABLE
        wells = [
            '100_01-04-081-24W4_00', '100_01-09-082-25W4_00', '100_01-26-081-25W4_00',
            '100_01-28-081-25W4_00', '100_02-36-081-25W4_00', '100_03-09-081-25W4_00'
        ]
        
        np.random.seed(42)
        for i, well in enumerate(wells):
            # Extract township/range info
            parts = well.split('_')[1].split('-')
            if len(parts) >= 4:
                # Simple coordinate generation based on location info
                section = float(parts[1])
                township = float(parts[2])
                
                # Create a grid pattern but add some jitter
                x_base = i % 3 * 1000
                y_base = i // 3 * 1000
                
                # Add a row for this well
                map_data.append({
                    'well_name': well,
                    'x_coord': x_base + section * 10 + np.random.normal(0, 10),
                    'y_coord': y_base + township * 10 + np.random.normal(0, 10),
                    'formation': 'McMurray',
                    'net_pay': np.random.uniform(5, 20),
                    'avg_phie': np.random.uniform(0.08, 0.18),
                    'avg_sw': np.random.uniform(0.3, 0.6),
                    'hydrocarbon_pore_volume': np.random.uniform(50000, 200000),
                    'recoverable_oil': np.random.uniform(10000, 80000)
                })
        
        map_df = pd.DataFrame(map_data)
        map_file = os.path.join(self.dirs['pay_analysis'], "pay_mapping_data.csv")
        map_df.to_csv(map_file, index=False)
        print(f"Created pay mapping data with {len(map_data)} points")
        
        return self.dirs['pay_analysis']
    
    def create_project_summary(self):
        """Create a project summary dashboard and integrate all data components"""
        
        print(f"Creating project summary...")
        
        # Count files in each subdirectory
        dir_counts = {}
        for subdir_name, subdir_path in self.dirs.items():
            if os.path.exists(subdir_path):
                file_count = len(glob.glob(os.path.join(subdir_path, "*.*")))
                dir_counts[subdir_name] = file_count
        
        # Create summary stats
        summary_stats = []
        
        # Well count
        well_index_path = os.path.join(self.base_dir, "master_well_index.csv")
        if os.path.exists(well_index_path):
            wells_df = pd.read_csv(well_index_path)
            well_count = len(wells_df)
            summary_stats.append({"category": "Wells", "count": well_count, "status": "Ready"})
            
            # Quality breakdown
            if 'data_quality' in wells_df.columns:
                quality_counts = wells_df['data_quality'].value_counts().to_dict()
                for quality, count in quality_counts.items():
                    summary_stats.append({"category": f"{quality} Quality Wells", "count": count, "status": "Analyzed"})
        else:
            summary_stats.append({"category": "Wells", "count": 0, "status": "Missing"})
        
        # Log curves
        curve_path = os.path.join(self.base_dir, "log_curve_inventory.csv")
        if os.path.exists(curve_path):
            curves_df = pd.read_csv(curve_path)
            curve_count = len(curves_df)
            unique_curve_count = curves_df['curve_name'].nunique() if 'curve_name' in curves_df.columns else 0
            summary_stats.append({"category": "Log Curves", "count": curve_count, "status": "Ready"})
            summary_stats.append({"category": "Unique Curve Types", "count": unique_curve_count, "status": "Ready"})
        else:
            summary_stats.append({"category": "Log Curves", "count": 0, "status": "Missing"})
        
                # Directory file counts
        for dirname, count in dir_counts.items():
            summary_stats.append({"category": f"{dirname} Files", "count": count, "status": "Ready" if count > 0 else "Empty"})
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary
        summary_file = os.path.join(self.dirs['summary'], "project_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Created project summary at: {summary_file}")
        
        # Create project status report
        status_file = os.path.join(self.dirs['summary'], "project_status.txt")
        with open(status_file, "w") as f:
            f.write("PETROPHYSICAL ANALYSIS PROJECT STATUS\n")
            f.write("===================================\n\n")
            
            f.write(f"Project Directory: {self.base_dir}\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PROJECT COMPONENTS STATUS\n")
            f.write("-------------------------\n")
            for _, row in summary_df.iterrows():
                f.write(f"{row['category']}: {row['count']} ({row['status']})\n")
            
            f.write("\n\nNEXT STEPS\n")
            f.write("----------\n")
            f.write("1. Import real LAS data into the data directory\n")
            f.write("2. Run log normalization to standardize curve names and units\n")
            f.write("3. Calculate basic petrophysical properties (VSH, PHIE, SW)\n")
            f.write("4. Identify pay zones using the provided templates\n")
            f.write("5. Generate pay maps and opportunity matrices\n")
        
        print(f"Created project status report at: {status_file}")
        
        # Create lithology computation guide
        guide_file = os.path.join(self.dirs['summary'], "lithology_computation_guide.txt")
        with open(guide_file, "w") as f:
            f.write("LITHOLOGY COMPUTATION GUIDE\n")
            f.write("=========================\n\n")
            
            f.write("1. DATA PREPARATION\n")
            f.write("------------------\n")
            f.write("- Ensure all wells have standardized GR, RHOB, NPHI, and PE curves\n")
            f.write("- Quality control the input data to remove spikes and bad hole sections\n")
            f.write("- Normalize the log data across all wells\n\n")
            
            f.write("2. MINERAL MODEL SETUP\n")
            f.write("---------------------\n")
            f.write("- Define end-member minerals for the study area (Matrix properties.csv)\n")
            f.write("- Sandstone: Quartz (2.65 g/cc, -0.03 v/v, 1.8 PE)\n")
            f.write("- Limestone: Calcite (2.71 g/cc, 0.0 v/v, 5.1 PE)\n")
            f.write("- Dolomite: Dolomite (2.87 g/cc, 0.02 v/v, 3.1 PE)\n")
            f.write("- Shale: Illite or mixed clay (2.53 g/cc, 0.30 v/v, 3.5 PE)\n")
            f.write("- Anhydrite: (2.98 g/cc, 0.0 v/v, 5.0 PE)\n\n")
            
            f.write("3. MINERAL COMPUTATION METHODS\n")
            f.write("----------------------------\n")
            f.write("- Deterministic approach: Use simultaneous equations to solve for mineral volumes\n")
            f.write("- Probabilistic approach: Use Bayesian inversion methods\n")
            f.write("- Machine Learning: Use cluster analysis or neural nets for lithofacies identification\n\n")
            
            f.write("4. RECOMMENDED TECHLOG WORKFLOW\n")
            f.write("-----------------------------\n")
            f.write("- Import logs into Techlog\n")
            f.write("- Create a Multi-Mineral Processing module\n")
            f.write("- Define mineral end-members and fluid properties\n")
            f.write("- Set up constraints (total volume = 1.0, non-negative volumes)\n")
            f.write("- Run the inversion\n")
            f.write("- QC results with core data if available\n")
        
        print(f"Created lithology computation guide at: {guide_file}")
        
        # Create guide for pay mapping
        pay_guide_file = os.path.join(self.dirs['summary'], "pay_mapping_guide.txt")
        with open(pay_guide_file, "w") as f:
            f.write("PAY MAPPING AND BYPASS PAY IDENTIFICATION GUIDE\n")
            f.write("============================================\n\n")
            
            f.write("1. DEFINE PAY CRITERIA\n")
            f.write("-------------------\n")
            f.write("- Use the pay_cutoffs.csv file to define cutoffs for each formation\n")
            f.write("- Typical cutoffs:\n")
            f.write("  * VSH < 0.4 (cleaner reservoir)\n")
            f.write("  * PHIE > 0.08 (sufficient porosity)\n")
            f.write("  * SW < 0.6 (hydrocarbon presence)\n")
            f.write("  * K > 1-10 mD (formation-dependent)\n\n")
            
            f.write("2. FLAG PAY ZONES\n")
            f.write("---------------\n")
            f.write("- Calculate flag curves in your log analysis workflow\n")
            f.write("- Generate summations of net pay for each well/zone\n")
            f.write("- Populate the pay_analysis_template.csv with real data\n\n")
            
            f.write("3. IDENTIFY BYPASS PAY\n")
            f.write("--------------------\n")
            f.write("- Compare identified pay intervals with perforation history\n")
            f.write("- Flag unperforated pay intervals\n")
            f.write("- Calculate potential hydrocarbon volumes\n")
            f.write("- Rank opportunities based on quality and thickness\n\n")
            
            f.write("4. CREATE PAY MAPS\n")
            f.write("-----------------\n")
            f.write("- Use the pay_mapping_data.csv template for data organization\n")
            f.write("- Import to mapping software (Petrel, Techlog Maps, etc.)\n")
            f.write("- Generate contour maps for:\n")
            f.write("  * Net Pay Thickness\n")
            f.write("  * Average Porosity\n")
            f.write("  * Hydrocarbon Pore Volume\n")
            f.write("  * Recoverable Reserves\n")
            f.write("- Identify sweet spots and bypass opportunities\n")
        
        print(f"Created pay mapping guide at: {pay_guide_file}")
        
        return self.dirs['summary']
    
    def create_las_standardization_template(self):
        """Create a template for standardizing LAS file headers"""
        
        print("Creating LAS standardization template...")
        
        # Define standard curve mnemonics mapping
        curve_standards = {
            # Standard name: [list of possible aliases]
            'DEPTH': ['DEPT', 'MD', 'DEPTH', 'TVD', 'DEP'],
            'GR': ['GR', 'GAMMA', 'GRGC', 'GRC', 'GRD'],
            'RHOB': ['RHOB', 'DENB', 'DEN', 'RHOZ'],
            'NPHI': ['NPHI', 'NEU', 'NPOR', 'TNPH'],
            'RT': ['RT', 'RD', 'RESD', 'RLA5', 'ILD', 'LLD'],
            'RXOZ': ['RXOZ', 'RS', 'RESS', 'RLA1', 'ILS', 'LLS'],
            'PEF': ['PEF', 'PE'],
            'CALI': ['CALI', 'CAL', 'HDIA', 'CALIPER'],
            'DT': ['DT', 'SONIC', 'DTCO', 'DELT'],
            'SP': ['SP', 'SPONP'],
            'VSH': ['VSH', 'VSHALE', 'SHALE', 'VSH_GR'],
            'PHIE': ['PHIE', 'PHIN', 'PORE', 'PORZ', 'PHIS']
        }
        
        # Define standard units
        standard_units = {
            'DEPTH': 'M',
            'GR': 'GAPI',
            'RHOB': 'G/C3',
            'NPHI': 'V/V',
            'RT': 'OHMM',
            'RXOZ': 'OHMM',
            'PEF': 'B/E',
            'CALI': 'IN',
            'DT': 'US/F',
            'SP': 'MV',
            'VSH': 'V/V',
            'PHIE': 'V/V'
        }
        
        # Create standardization template
        template = {
            'curve_mapping': curve_standards,
            'units': standard_units,
            'header_template': {
                'COMP': 'Your Company',
                'DATE': 'Auto',  # Will be filled with current date
                'STEP': 0.1,     # Default step size in meters
                'NULL': -999.25,  # Standard null value
                'SRVC': 'VARIOUS'
            }
        }
        
        # Save as a text file for reference
        template_file = os.path.join(self.base_dir, "las_standardization_template.txt")
        
        with open(template_file, 'w') as f:
            f.write("LAS STANDARDIZATION TEMPLATE\n")
            f.write("============================\n\n")
            
            f.write("CURVE MAPPINGS:\n")
            for std_name, aliases in curve_standards.items():
                f.write(f"{std_name}: {', '.join(aliases)}\n")
            
            f.write("\nSTANDARD UNITS:\n")
            for curve, unit in standard_units.items():
                f.write(f"{curve}: {unit}\n")
            
            f.write("\nHEADER TEMPLATE:\n")
            for key, value in template['header_template'].items():
                f.write(f"{key}: {value}\n")
        
        print(f"LAS standardization template created at: {template_file}")
        
        return template_file
# Add the check_dependencies method around line 722-725
def check_dependencies(self):
    """Check if required libraries are installed"""
    
    print("Checking required dependencies...")
    
    # Define required libraries
    required_libs = {
        'pandas': 'For data manipulation and CSV handling',
        'numpy': 'For numerical operations',
        'matplotlib': 'For visualization and charts (optional but recommended)',
        'lasio': 'For LAS file handling (optional for initial setup)'
    }
    
    # Check required libraries
    missing_libs = []
    for lib, purpose in required_libs.items():
        try:
            importlib.import_module(lib)
            print(f"✓ {lib} is installed - {purpose}")
        except ImportError:
            print(f"✗ {lib} is NOT installed - {purpose}")
            missing_libs.append(lib)
    
    # Handle missing libraries
    if missing_libs and ('pandas' in missing_libs or 'numpy' in missing_libs):
        print("\nCritical libraries are missing. Please install them with:")
        print(f"pip install {' '.join(missing_libs)}")
        return False
    elif missing_libs:
        print("\nSome recommended libraries are missing. For full functionality, install:")
        print(f"pip install {' '.join(missing_libs)}")
        return True  # Non-critical libraries missing
    else:
        print("\nAll required libraries are installed!")
        return True

    def run_complete_workflow(self):
        """Run the complete workflow for petrophysical analysis setup"""
        print("\n========== RUNNING COMPLETE PETROPHYSICAL ANALYSIS WORKFLOW ==========\n")
        
        # Check dependencies first
        if not self.check_dependencies():
            print("Critical dependencies missing. Please install required packages.")
            return None              
        
        # Step 1: Create master well index
        print("\n--- STEP 1: Creating Master Well Index ---")
        self.create_master_well_index()
        
        # Step 2: Create log curve inventory
        print("\n--- STEP 2: Creating Log Curve Inventory ---")
        self.create_log_inventory()
        
        # Step 3: Assess data quality
        print("\n--- STEP 3: Assessing Data Quality ---")
        self.assess_data_quality()
        
        # Step 4: Prepare lithology data
        print("\n--- STEP 4: Preparing Lithology Data ---")
        self.prepare_lithology_data()
        
        # Step 5: Prepare pay analysis
        print("\n--- STEP 5: Preparing Pay Analysis ---")
        self.prepare_pay_analysis()
        
        # Step 6: Create standardization template
        print("\n--- STEP 6: Creating LAS Standardization Template ---")
        self.create_las_standardization_template()
        
        # Step 7: Create project summary
        print("\n--- STEP 7: Creating Project Summary ---")
        self.create_project_summary()
        
        print("\n========== WORKFLOW COMPLETE ==========\n")
        print(f"Project location: {self.base_dir}")
        print("You can now begin importing real data and performing analysis.")
        
        return self.base_dir


# Execute the workflow if run as a script
if __name__ == "__main__":
    # Initialize the workflow with a default directory
    workflow = PetrophysicalWorkflow()
    
    # Run the complete workflow
    project_dir = workflow.run_complete_workflow()
