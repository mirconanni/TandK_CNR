########## IMPORT LIBRARIES ########## 

# sys is required to use the open function to write on file
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pickle
import folium
from folium import plugins
from folium.plugins import HeatMap
import selenium.webdriver
import json
from sklearn import preprocessing
import bezier
import colorlover as cl

plt.rcParams["font.family"] = 'serif'


########## IMPORT MY SCRIPTS ########## 

from compute_stops import read_params


########## FUNCTION DEFINITION ##########ù

#
def save_map(v, i, id_area, m, cluster_on):
    file_name = '../../thesis/images/geo_imn_area'+id_area+"_v"+i+"_"+v
    if cluster_on:
        file_name += "_clust"
    m.save(file_name+'.html')
    
    driver = selenium.webdriver.PhantomJS()
    driver.set_window_size(2500, 1800)
    driver.get(file_name+'.html')
    driver.save_screenshot(file_name+'.png')
    return 0

#
def get_bearing(p1, p2):
    '''
    Returns compass bearing from p1 to p2

    Parameters
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon

    Return
    compass bearing of type float

    Notes
    Based on https://gist.github.com/jeromer/2005586
    '''

    long_diff = np.radians(p2[0] - p1[0])

    lat1 = np.radians(p1[1])
    lat2 = np.radians(p2[1])

    x = np.sin(long_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2)
         - (np.sin(lat1) * np.cos(lat2)
            * np.cos(long_diff)))
    bearing = np.degrees(np.arctan2(x, y))

    # adjusting for compass bearing
    if bearing < 0:
        return bearing + 360
    return bearing

#
def draw_edges_map(fmov, weight, m):
    for i, fm in enumerate(fmov):
        folium.PolyLine(fm, color="#000066", weight=weight[i], opacity=0.8).add_to(m)
        s, e = fm[0], fm[-2]
        #rotation = get_bearing(s, e) - 90
        #folium.RegularPolygonMarker(location=e, color="#000066", fill=True, fill_color="#FFFFFF",
        #                                    fill_opacity=0.8, number_of_sides=3, radius=6, rotation=rotation).add_to(m)

#
def compute_edge_shape(location_nextlocs, location_prototype):
    fmov = list()
    weight = list()
    for lid1 in location_nextlocs:
        for lid2 in location_nextlocs[lid1]:
            s = location_prototype[lid1]
            e = location_prototype[lid2]
            gap = 0.05 * abs(e[1] - s[1]) / 0.05
            nodes = np.asfortranarray([
                [s[1], (s[1] + e[1]) / 2 + np.random.choice([gap, -gap]), e[1]],
                [s[0], (s[0] + e[0]) / 2 + np.random.choice([gap, -gap]), e[0]],
            ])
            curve = bezier.Curve(nodes, degree=2)
            val = curve.evaluate_multi(np.linspace(0.0, 1.0, 10))
            x_val = val[0]
            y_val = val[1]
            mov = list()
            for xv, yv in zip(x_val, y_val):
                mov.append([xv, yv])

            fmov.append(mov)
            weight.append(np.log(location_nextlocs[lid1][lid2] * 10))
    return fmov, weight

#
def get_3_vehicles(path, file_name_out, id_area):
     # read the df containing the purity and the entropy of the clusters
    df_purity = pd.read_csv(path+"df_purity"+file_name_out+'.csv')

    vehicles = []
    v1 = df_purity.loc[df_purity["purity_p"] > 0.2]
    v1 = v1.loc[v1["purity_p"] < 0.3]
    v1 = v1.loc[v1["entropy_p"] > 0.9]
    v1 = v1.loc[v1["tot_loc"] > 80]
    v1 = v1.loc[v1["tot_loc"] < 180]
    vehicles.append(v1.iloc[0]["vehicle"])

    v2 = df_purity.loc[df_purity["purity_p"] > 0.5]
    v2 = v2.loc[v2["purity_p"] < 0.6]
    v2 = v2.loc[v2["entropy_p"] > 0.5]
    v2 = v2.loc[v2["entropy_p"] < 0.6] 
    v2 = v2.loc[v2["tot_loc"] > 80]
    v2 = v2.loc[v2["tot_loc"] < 180]

    vehicles.append(v2.iloc[0]["vehicle"])

    if id_area == "2":
        v3 = df_purity.loc[df_purity["purity_p"] > 0.9]
        v3 = v3.loc[v3["entropy_p"] < 0.3] 
        v3 = v3.loc[v3["tot_loc"] > 80]
    else:
        v3 = df_purity.loc[df_purity["purity_p"] > 0.8]
        v3 = v3.loc[v3["entropy_p"] < 0.35] 
        v3 = v3.loc[v3["tot_loc"] > 60]
    
    vehicles.append(v3.iloc[0]["vehicle"])
    print(vehicles)

    return vehicles

#
def visualize_imn(stop, id_area, month_code, week, vis_traj = True, tiles='stamentoner', cluster_on=True):
    # get dataframe not normalized
    path = '../../datasets/out/Traj' + stop + 'min/'
    file_name = '_area'+id_area+'_month'+month_code+'_week'+ week
    file_name_in = 'loc_feat'+ file_name + '_complete.csv'

    # read dataframe with location features, extract only the ones we care about
    df_complete = pd.read_csv(path+file_name_in)
    df_complete = df_complete[["vehicle", "loc_id", "loc_proto_lat", "loc_proto_lon", "support"]]

    # read the list of linkage clusters for each location
    with open(path + "link_cluster" + file_name + '_log.pickle', 'rb') as fp:
        df_link = pickle.load(fp)
        link_cluster = df_link["link_cluster"]

    # store the linkage cluster as new column of the df
    df_complete = df_complete.assign(link_cluster = link_cluster) 

    # extract 3 vehicle with different purity levels
    vehicles = get_3_vehicles(path, file_name + '_log', id_area)

    for ix, v in enumerate(vehicles):

        # extract the df of locations only of that vehicle
        df_v = df_complete[df_complete["vehicle"] == v]
        
        # read the imn of the vehicle
        with open(path+"imn_light"+file_name+'.json', 'rb') as fp:
            file_j = json.load(fp)
            imn_v = file_j[v]
        
        # extract the dict of nextlocations and the location features
        location_nextlocs = imn_v["location_nextlocs"]

        # compute a list of points coordinates
        points = np.array([[p[0], p[1]] for p in zip(df_v["loc_proto_lon"], df_v["loc_proto_lat"])])

        # compute a dict from the location id (as string) to its coordinates
        loc_id_string = [str(x) for x in df_v["loc_id"]]
        location_prototype = dict(zip(loc_id_string, points))

        # get map center location and zoom start according to the area
        if id_area == "2":
            c_lat, c_lon = [38.25, 23.4]
            zoom_start = 11
        else:
            c_lat, c_lon = [38, 23.68]
            zoom_start = 12

        # draw the map
        m = folium.Map(location=[c_lat, c_lon], zoom_start=zoom_start, tiles=tiles)
            
        # if you want to draw also the trajectories
        if vis_traj:
            # compute the shape and the weight of the edges
            fmov, weight = compute_edge_shape(location_nextlocs, location_prototype) 

            #for each edge append to the map a line and a marker to show the direction
            draw_edges_map(fmov, weight, m)

        lat_list = list(df_v["loc_proto_lat"])
        lon_list = list(df_v["loc_proto_lon"])
        sup_list = list(df_v["support"])

        sup_list = [np.sqrt(x * 10000) for x in sup_list]

        if cluster_on:
            # nero, giallo, rosso, verde, blu, viola
            colors = ["#525252", "#ffff33", "#e31a1c", "#33a02c", "#1f78b4", "#e7298a"]
            link_cluster = df_v["link_cluster"]
            #print(list(link_cluster))
            sup_colors = [colors[l-1] for l in link_cluster]

            # dic = dict()
            # for ki,vi in zip (link_cluster, list(df_v["support"])):
            #     if ki not in dic:
            #         dic[ki] = 0
            #     dic[ki] += vi


            # print(dic)

        else:
            q=np.array([0.0, 0.25, 0.50, 0.75, 1.0])        
            sup_colors = pd.qcut(sup_list, q=q, duplicates='drop')
            colors = list(cl.scales['9']['seq']['Blues'])[9 - len(sup_colors.categories):]
            sup_colors = pd.qcut(sup_list, q=q, labels=colors, duplicates='drop')

        for i in range(0, len(lon_list)):
            folium.Circle(location=(lat_list[i], lon_list[i]), radius=sup_list[i], color=sup_colors[i], fill=True,
                        fill_color=sup_colors[i], fill_opacity=0.8).add_to(m)

        title_html = """<div style="position: fixed; 
                    top: 20px; left: 50px; width: 800px; height: 90px; 
                    z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Individual Mobility Network</div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        save_map(v, str(ix+1), id_area, m, cluster_on)

    return 0

########## MAIN FUNCTION ##########

def main():

    stop, id_area, month_code, week = read_params(sys)

    visualize_imn(stop, id_area, month_code, week)
    
    return 0


if __name__ == "__main__":
    main()