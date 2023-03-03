import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import utilities as utils
import configuration as config
import datacommons_pandas as dc

def add_name_geoId(df):
    
  # add a new column called name, 
  # where each value is the name for the place dcid in the index
  df['name'] = df.index.map(dc.get_property_values(df.index, 'name'))
  df['geoId'] = df.index.map(dc.get_property_values(df.index, 'geoId'))
  
  # keep just the first name, instead of a list of all names.
  df['name'] = df['name'].str[0]

  # geoId is a list of one element. convert to scalar
  df['geoId'] = df['geoId'].str[0]

# start the timer
start = time.perf_counter()

# data commons ID for United States is country/USA
usa = 'country/USA'

# get list of all counties in the US
counties = dc.get_places_in([usa], 'County')[usa]

# get StatVarObservations for counties
df_county = dc.build_multivariate_dataframe(counties, \
    ['Median_Income_Person', 'Count_HousingUnit', 'Count_Person'])

# add the county name and geoId
add_name_geoId( df_county )

# read the naics file
naicsData = pd.read_csv( config.naics, dtype=str )

# find all the geopackage files
geoFiles = utils.getListOfFiles( config.dataDir )
print("Found", len(geoFiles), "geopackage files")

# counters
ix = 0
buildingCount = 0
trianingCount = 0
percent10 = np.ceil(0.10 * len(geoFiles))
percent25 = np.ceil(0.25 * len(geoFiles))
percent50 = np.ceil(0.50 * len(geoFiles))
percent75 = np.ceil(0.75 * len(geoFiles))
percent90 = np.ceil(0.90 * len(geoFiles))

# open the output files
stateFips = '37'
trainingFile = 'ML_Training_' + stateFips + '.csv'

# without HAZUS summary
header = "StateFips,CountyFips,StateCountyFips," + \
    "X,Y,Area,MedianIncomeCounty,HousingUnitsCounty," + \
    "HousingDensityCounty,Impervious,AgCount,CmCount,GvCount,EdCount," + \
    "InCount,OsmNearestRoad,OrnlType\n"

#######
# building height (ORNL) is not available very often, ignored for ML
#######

outFile = open( config.outDir + trainingFile, 'w' )
outFile.write(header)
outFile.close()   

# open another file for all unknown buildings
header = "StateFips,CountyFips,StateCountyFips," + \
    "X,Y,Area,MedianIncomeCounty,HousingUnitsCounty," + \
    "HousingDensityCounty,Impervious,AgCount,CmCount,GvCount,EdCount," + \
    "InCount,OsmNearestRoad,FEMA_100yr\n"
unknownFile = open( config.outDir + 'uknownBuildings.csv', 'w' )
unknownFile.write(header)

# counters for ORNL types
ornl = { 'Agriculture':0, 'Commercial':0, 'Residential':0, 'Education':0,
    'Government':0, 'Industrial':0, 'Utility and Misc':0, 'Assembly':0, 'Unclassified':0 }

# iterate over all the geopackage files
usableBuildingsPerCounty = {}
for file in geoFiles:

    if ( ix == percent10 ):
        print('10% complete...')
    if ( ix == percent25 ):
        print('25% complete...')
    if ( ix == percent50 ):
        print('50% complete...')
    if ( ix == percent75 ):
        print('75% complete...')
    if ( ix == percent90 ):
        print('90% complete...')

    # extract all the buildings
    data = gpd.read_file(file)
    buildings = data[ data['osm_building'] == 'yes' ]
    buildingCount += buildings.shape[0]

    # valid buildings are those with known type to be used in supervised learning
    validBuildings = buildings.loc[ buildings['ornl_OCC_CLS'].notnull() ]

    # unknown buildings are those without known type
    unknownBuildings = buildings.loc[ buildings['ornl_OCC_CLS'].isnull() ] 
    
    n1 = validBuildings.shape[0]
    n2 = unknownBuildings.shape[0]
    n3 = buildings.shape[0]
    assert (n1+n2) == n3, "building shapes not correct"
    
    # count of valid buildings found
    usableBuildingsThisCounty = validBuildings.shape[0]
    
    # get fips codes for this county
    stateValues = validBuildings.loc[ validBuildings['STATEFP'].notnull() ] # values are occasionally None
    countyValues = validBuildings.loc[ validBuildings['COUNTYFP'].notnull() ]
    state = stateValues['STATEFP'].iloc[0] 
    county = countyValues['COUNTYFP'].iloc[0] 
    print("Working on state/county:", state, county)
    print("   Usable buildings:", usableBuildingsThisCounty )

    # get the naics summary for this county
    agN, cmN, gvN, edN, idN = utils.getNaicsSummary( naicsData, state, county )

    # open the state file
    outState = open( config.outDir + trainingFile, 'a' )

    # keep track of usable buildings per county
    usableBuildingsPerCounty[ state+county ] = usableBuildingsThisCounty

    # get county data from data commons
    medianIncomeCounty, housingUnitsCounty, housingDensityCounty = \
        utils.getCountyData( state+county, df_county )

    # iterate over all the unknown buildings
    for idex, row in unknownBuildings.iterrows():
        
        # get values from data file
        x = row['x']
        y = row['y']
        s = gpd.GeoSeries( row['geometry'], crs='epsg:5070' )
        area = float(s.area) * 1.22e10 
        imp = row['imperv']
        #height = row['ornl_HEIGHT']
        nearestRoad = row['osm_nearest_road_type']
        secondaryType = row['ornl_PRIM_OCC']
        fema = row['fema_100yr']
            
        # output line
        line = state + ',' + county + ',' + state+county + ',' + \
            str(x) + ',' + str(y) + ',' + str(area) + ',' + str(medianIncomeCounty) + ',' + \
            str(housingUnitsCounty) + ',' + str(housingDensityCounty) + ',' + \
            str(imp) + ',' + str(agN) + ',' + str(cmN) + ',' + \
            str(gvN) + ',' + str(edN) + ',' + str(idN) + ',' + \
            str(nearestRoad) + ',' + str(fema) + '\n'
        
        unknownFile.write( line )
        
    # iterate over all the valid buildings    
    for idex, row in validBuildings.iterrows():

        # get ORNL OCC type, which are the same names as HAZUS
        ornlType = row['ornl_OCC_CLS']
        cnt = ornl[ ornlType ]
        ornl[ ornlType ] = cnt + 1

        # continue if usable ORNL type
        if ( (ornlType != None) and (ornlType != 'Unclassified') ):

            # update the training size counter
            trianingCount += 1
        
            # get values from data file
            x = row['x']
            y = row['y']
            s = gpd.GeoSeries( row['geometry'], crs='epsg:5070' )
            area = float(s.area) * 1.22e10 
            imp = row['imperv']
            #height = row['ornl_HEIGHT']
            nearestRoad = row['osm_nearest_road_type']
            secondaryType = row['ornl_PRIM_OCC']
            
            # output line
            line = state + ',' + county + ',' + state+county + ',' + \
             str(x) + ',' + str(y) + ',' + str(area) + ',' + str(medianIncomeCounty) + ',' + \
             str(housingUnitsCounty) + ',' + str(housingDensityCounty) + ',' + \
             str(imp) + ',' + str(agN) + ',' + str(cmN) + ',' + \
             str(gvN) + ',' + str(edN) + ',' + str(idN) + ',' + \
             str(nearestRoad) + ',' + ornlType + '\n'
        
            outState.write( line )

# close the state file
outState.close()

# close the unknown buildings file
unknownFile.close()

# output usable buildings per county
outFile = open( config.outDir + 'ML_UsableBuildingsPerCounty.csv', 'w' )
outFile.write("StateCounty,Buildings\n")
for key, value in usableBuildingsPerCounty.items():
    outFile.write( str(key) + ',' + str(value) + '\n' )
outFile.close

# output counts per ORNL type
outFile = open( config.outDir + 'ML_HazusCounts.csv', 'w' )
outFile.write("Type,Count\n")
for key, value in ornl.items():
    outFile.write( str(key) + ',' + str(value) + '\n' )
outFile.close

# output statistics
outFile = open( config.outDir + 'ML_Data_Summary.txt', 'w' )
outFile.write("Total number of buildings: " + str(buildingCount) + "\n" )
outFile.write("Buildings in training set: " + str(trianingCount) + "\n" )

# stop the timer
stop = time.perf_counter()
duration = stop - start # in seconds
duration /= 60 # in minutes
duration /= 60 # in hours
outFile.write("Code took " + str(np.round(duration,2)) + " hours to run\n")
outFile.close()
