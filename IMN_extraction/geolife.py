import glob
import pandas as pd

from IMN_extraction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from IMN_extraction.imn_extractor import imn_extract

def main():
    mypath = "Geolife Trajectories 1.3/Data/064/Trajectory/"
    uid = 64
    all_points = []
    for f in sorted(glob.glob(mypath + "*.plt")):
        points = pd.read_csv(f, skiprows=6, header=None, parse_dates=[[5, 6]])
        for i,p in points.iterrows():
            all_points.append({"RECORD_TYPE": 'P',
                               "VEHICLE_ID": uid,
                               "TIMESTAMP": p['5_6'],
                               "location": {"lat": p[0], "lon": p[1]}
                               })

    mongo_connector = TaK_Mongo_Connector('localhost', '27017', 'test')
    mongo_connector.insert_many('geolife_data', all_points)
    mongo_connector.create_index('geolife_data', 'TIMESTAMP', True)

    imn_extract('localhost', '27017', 'test', 'geolife_data', 'geolife_imns', 'geolife_users.txt', True, 0.07, 0.05, 1200, '2008-08-15T00:00:00.000', '2008-08-31T00:00:00.000', 5, 1.0, 60)





if __name__ == "__main__":
    main()