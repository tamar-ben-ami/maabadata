DATA_PATH = r"C:\Users\Micha\Documents\HUJI\Competitive\Project\MapProject\Data"
WEATHER_DATA_PATH = DATA_PATH + r"\WeatherNormal"

LAYERS = {"electricity_transition_lines": r"Electricity\Electric__Power_Transmission_Lines_WGS84.shp",
          "electricity_power_plants": r"Electricity\Power_Plants_WGS84.shp",
          "indian_reserves": r"IndianReserves\tl_2018_us_aiannh_WGS84.shp",
          "military_bases": r"Military\Military_Bases_WGS84.shp",
          "national_parks": r"NationalParks\NPS_-_Land_Resources_Division_Boundary_and_Tract_Data_Service_WGS84.shp",
          "transportation_amtrak_stations": r"Transportation\Amtrak_Stations_WGS84.shp",
          "transportation_rail_network_lines": r"Transportation\North_American_Rail_Network_Lines_WGS84.shp",
          "transportation_rail_network_nodes": r"Transportation\North_American_Rail_Network_Nodes_WGS84.shp",
          "transportation_railroad_bridges": r"Transportation\Railroad_Bridges_WGS84.shp",
          "transportation_railroad_grade_crossings": r"Transportation\Railroad_Grade_Crossings_WGS84.shp",
          "water_bodies": r"Water\USA_Detailed_Water_Bodies_WGS84.shp",
          "urban_areas": r"UrbanAreas\tl_2019_us_uac10_WGS84.shp"}

# weather rasters
ppt_dir_name = "Percipitation"
tmean_dir_name = "MeanTemperature"
tmin_dir_name = "MinTemperature"
tmax_dir_name = "MaxTemperature"
tdmean_dir_name = "MeanDewTemperature"
vpdmin_dir_name = "MinVaporPressureDeficit"
vpdmax_dir_name = "MaxVaporPressureDeficit"
solclear_dir_name = "SolarRadiationClear"
soltotal_dir_name = "SolarRadiationHorizontal"
solslope_dir_name = "SolarRadiationSloped"
soltrans_dir_name = "CloudTransmittance"

ppt_file_name = "PRISM_ppt_30yr_normal_4kmM4_{}_bil.bil"
tmean_file_name = "PRISM_tmean_30yr_normal_4kmM4_{}_bil.bil"
tmin_file_name = "PRISM_tmin_30yr_normal_4kmM4_{}_bil.bil"
tmax_file_name = "PRISM_tmax_30yr_normal_4kmM4_{}_bil.bil"
tdmean_file_name = "PRISM_tdmean_30yr_normal_4kmM4_{}_bil.bil"
vpdmax_file_name = "PRISM_vpdmax_30yr_normal_4kmM4_{}_bil.bil"
vpdmin_file_name = "PRISM_vpdmin_30yr_normal_4kmM4_{}_bil.bil"
solclear_file_name = "PRISM_solclear_30yr_normal_4kmM3_{}_bil.bil"
soltotal_file_name = "PRISM_soltotal_30yr_normal_4kmM3_{}_bil.bil"
solslope_file_name = "PRISM_solslope_30yr_normal_4kmM3_{}_bil.bil"
soltrans_file_name = "PRISM_soltrans_30yr_normal_4kmM3_{}_bil.bil"

WEATHER_FEATURES_MAP = {"precipitation": ppt_dir_name,
                        "mean_temp": tmean_dir_name,
                        "min_temp": tmin_dir_name,
                        "max_temp": tmax_dir_name,
                        "mean_dew_temp": tdmean_dir_name,
                        "min_vapor_pressure": vpdmin_dir_name,
                        "max_vapor_pressure": vpdmax_dir_name,
                        "solar_radiation_clear": solclear_dir_name,
                        "solar_radiation_horizontal": soltotal_dir_name,
                        "solar_radiation_slope": solslope_dir_name,
                        "cloud_transmittance": soltrans_dir_name}
WEATHER_FILES_MAP = {ppt_dir_name: ppt_file_name,
                     tmean_dir_name: tmean_file_name,
                     tmin_dir_name: tmin_file_name,
                     tmax_dir_name: tmax_file_name,
                     tdmean_dir_name: tdmean_file_name,
                     vpdmin_dir_name: vpdmin_file_name,
                     vpdmax_dir_name: vpdmax_file_name,
                     solclear_dir_name: solclear_file_name,
                     soltotal_dir_name: soltotal_file_name,
                     solslope_dir_name: solslope_file_name,
                     soltrans_dir_name: soltrans_file_name}
