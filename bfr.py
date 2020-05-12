# python3 bfr.py <input_path> <n_cluster> <out_file1> <out_file2>
# python3 bfr.py test1 10 output1.json intermediate1.csv
# test1-10, test2-10, test3-5, test4-8, test5-15
# test1 5 * 137923 = 689617 192 0,
# test2 10 * 21318 = 213184 947 99,
# test3 3 * 29609 = 80728 1899 10,
# test4 6 * 34643 = 207862 564 30,
# test5 8 * 33848 = 270799 189 20


# from pyspark import SparkContext
import sys
import os
import re
import random
import time
import math
import copy
from collections import Counter
import json
import statistics


start_time = time.time()

input_path = sys.argv[1]
n_cluster = int(sys.argv[2])
out_file1 = sys.argv[3]
out_file2 = sys.argv[4]

# os.environ['PYSPARK_PYTHON'] = 'usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = 'usr/local/bin/python3.6'
#
# sc = SparkContext('local[*]', 'bfr')
# sc.setLogLevel('ERROR')


txt_paths = []
for root, dirs, files in os.walk(input_path):
    for filename in files:
        path = os.path.join(root, filename)
        txt_paths.append(path)


def p2cdisteu(p, c):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p, c)]))
    return distance


def p2cdistmht(p, c):
    distance = math.sqrt(sum([abs(a - b) for a, b in zip(p, c)]))
    return distance


def kmeanspp(pointlist, k):
    cts = []
    p0 = random.choice(pointlist)
    cts.append(p0)
    while len(cts) < k:
        max_dist = -1
        for point in pointlist:
            min_dist = float("inf")
            for ctd in cts:
                dist = p2cdisteu(point, ctd)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > max_dist:
                max_dist = min_dist
                nextselect = point.copy()
        cts.append(nextselect)
        # print(nextselect)
    return cts


def kmeans0(ipmap):
    # parameter
    pointnum = len(ipmap)
    its = dimension * pointnum * n_cluster
    if its < 1200000:
        maxiteration = 50
    elif its < 1800000:
        maxiteration = 40
    elif its < 2000000:
        maxiteration = 35
    else:
        maxiteration = 30
    if n_cluster < 8:
        maxcnumtimes = 1
    else:
        maxcnumtimes = 1
    # print(maxiteration, maxcnumtimes)
    # initialize
    # k means ++
    centroids = kmeanspp(list(ipmap.values()), k=min(math.ceil(n_cluster * maxcnumtimes), math.ceil(len(ipmap) / 2)))
    # print(centroids)
    # random.seed(0)
    # centroids = random.sample(list(ipmap.values()), k=min(math.ceil(n_cluster * maxcnumtimes), math.ceil(len(ipmap) / 2)))
    index_centroid_map = {}
    for index, centroid in enumerate(centroids):
        index_centroid_map[index] = centroid
    iteration = 0
    # parameter
    while iteration < maxiteration:
        centroid_points_dict = {}
        for indexI in ipmap:
            mindistance = float("inf")
            for indexJ in index_centroid_map:
                distance = p2cdistmht(ipmap[indexI], index_centroid_map[indexJ])
                if distance < mindistance:
                    mindistance = distance
                    cindex = indexJ
            if cindex not in centroid_points_dict:
                centroid_points_dict[cindex] = []
            centroid_points_dict[cindex].append(indexI)
        for index in centroid_points_dict:
            temppointlist = [ipmap[point_index] for point_index in centroid_points_dict[index]]
            # pointnum = len(temppointlist)
            newcentroid = [statistics.median(i) for i in zip(*temppointlist)]
            index_centroid_map[index] = newcentroid
        iteration += 1
    return centroid_points_dict


def p2cdistma(point, setinfodict):
    SUM = setinfodict['SUM']
    SUMSQ = setinfodict['SUMSQ']
    N = setinfodict['N']
    mulist = [x/N for x in SUM]
    sigmalist = [math.sqrt(SUMSQ[i]/N - (SUM[i]/N)**2) for i in range(len(SUM))]
    distance = math.sqrt(sum([((point[i]-mulist[i])/max(0.000001, sigmalist[i]))**2 for i in range(dimension)]))
    return distance


clusterdict = {}  # {data point index: cluster index}
discardsetinfodict = {}  # {cluster index: {N:,SUM:[],SUMSQ:[]}}
compressedsetinfodict = {}  # {cluster index: {N:,SUM:[],SUMSQ:[]}}
retainedsetdict = {}  # {data point index: [coordinates])

discard_cluster_points_dict = {}  # {cluster index: [point indices]}
compressed_cluster_points_dict = {}  # {cluster index: [point indices]}

intermediateoutput = open(out_file2, 'w')

indx = 1
for path in txt_paths:
    print(indx)
    f = open(path, 'r')
    lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    lines = [[line[0]] + [float(x) for x in line[1:]] for line in lines]
    dimension = len(lines[0])-1
    if indx == 1:
        index_point_map = {d[0]: d[1:] for d in lines}

        temppointlist0 = list(index_point_map.values())
        medianlist = [statistics.median(i) for i in zip(*temppointlist0)]
        stdevlist = [statistics.stdev(i) for i in zip(*temppointlist0)]

        ipmforkmeans = {}
        ipmforassign = {}
        for pid, pcoor in index_point_map.items():
            ifoutlier = 0
            for i in range(dimension):
                z_score = (pcoor[i] - medianlist[i]) / max(0.00001, stdevlist[i])
                if abs(z_score) > 10:
                    # print(z_score)
                    retainedsetdict[pid] = pcoor
                    ifoutlier = 1
                    break
            if ifoutlier == 0:
                if random.random() < 0.2:
                    ipmforkmeans[pid] = pcoor
                else:
                    ipmforassign[pid] = pcoor

        # print(retainedsetdict)

        centroid_points_dict = kmeans0(ipmforkmeans)


        for cid, pids in centroid_points_dict.items():
            for pid in pids:
                clusterdict[pid] = cid


        length_dict = {key: len(value) for key, value in centroid_points_dict.items()}
        tempddict = dict(Counter(length_dict).most_common(n_cluster))
        tempdindex = list(tempddict.keys())
        for cluster_index in tempdindex:
            for point_index in centroid_points_dict[cluster_index]:
                if cluster_index in discardsetinfodict:
                    discardsetinfodict[cluster_index]['N'] += 1
                    discardsetinfodict[cluster_index]['SUM'] = \
                        [sum(i) for i in zip(discardsetinfodict[cluster_index]['SUM'],
                                             index_point_map[point_index])]
                    discardsetinfodict[cluster_index]['SUMSQ'] = \
                        [sum(i) for i in zip(discardsetinfodict[cluster_index]['SUMSQ'],
                                             [x ** 2 for x in index_point_map[point_index]])]
                else:
                    discardsetinfodict[cluster_index] = {}
                    discardsetinfodict[cluster_index]['N'] = 1
                    discardsetinfodict[cluster_index]['SUM'] = index_point_map[point_index]
                    discardsetinfodict[cluster_index]['SUMSQ'] = \
                        [x ** 2 for x in index_point_map[point_index]]
                if cluster_index in discard_cluster_points_dict:
                    discard_cluster_points_dict[cluster_index].append(point_index)
                else:
                    discard_cluster_points_dict[cluster_index] = []
                    discard_cluster_points_dict[cluster_index].append(point_index)

        newdiscardsetinfodict = {}
        newdiscard_cluster_points_dict = {}
        for cindex, pindices in discard_cluster_points_dict.items():
            for pid in pindices:
                distance = p2cdistma(index_point_map[pid], discardsetinfodict[cindex])
                if distance < 4 * math.sqrt(dimension):
                    if cindex in newdiscardsetinfodict:
                        newdiscardsetinfodict[cindex]['N'] += 1
                        newdiscardsetinfodict[cindex]['SUM'] = \
                            [sum(i) for i in zip(newdiscardsetinfodict[cindex]['SUM'],
                                                 index_point_map[pid])]
                        newdiscardsetinfodict[cindex]['SUMSQ'] = \
                            [sum(i) for i in zip(newdiscardsetinfodict[cindex]['SUMSQ'],
                                                 [x ** 2 for x in index_point_map[pid]])]
                    else:
                        newdiscardsetinfodict[cindex] = {}
                        newdiscardsetinfodict[cindex]['N'] = 1
                        newdiscardsetinfodict[cindex]['SUM'] = index_point_map[pid]
                        newdiscardsetinfodict[cindex]['SUMSQ'] = \
                            [x ** 2 for x in index_point_map[pid]]
                    if cindex in newdiscard_cluster_points_dict:
                        newdiscard_cluster_points_dict[cindex].append(pid)
                    else:
                        newdiscard_cluster_points_dict[cindex] = []
                        newdiscard_cluster_points_dict[cindex].append(pid)
                else:
                    retainedsetdict[pid] = index_point_map[pid]
        discard_cluster_points_dict = newdiscard_cluster_points_dict.copy()
        discardsetinfodict = newdiscardsetinfodict.copy()

        for cluster_index, length in length_dict.items():
            if cluster_index not in tempdindex:
                if length > 0:
                    if length == 1:
                        for point_index in centroid_points_dict[cluster_index]:
                            point_coor = index_point_map[point_index]
                            assigned = 0
                            tempdict = {}
                            for cluster_index, cluster_info in discardsetinfodict.items():
                                distance = p2cdistma(point_coor, cluster_info)
                                tempdict[cluster_index] = distance
                            cminindex = min(tempdict, key=tempdict.get)
                            distance = tempdict[cminindex]
                            if distance < 4 * math.sqrt(dimension):
                                discard_cluster_points_dict[cminindex].append(point_index)
                                discardsetinfodict[cminindex]['N'] += 1
                                discardsetinfodict[cminindex]['SUM'] = \
                                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUM'],
                                                         point_coor)]
                                discardsetinfodict[cminindex]['SUMSQ'] = \
                                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUMSQ'],
                                                         [x ** 2 for x in point_coor])]
                                assigned = 1
                            if assigned == 0:
                                retainedsetdict[point_index] = point_coor
                    else:
                        for point_index in centroid_points_dict[cluster_index]:
                            point_coor = index_point_map[point_index]
                            assigned = 0
                            tempdict = {}
                            for cluster_index, cluster_info in discardsetinfodict.items():
                                distance = p2cdistma(point_coor, cluster_info)
                                tempdict[cluster_index] = distance
                            cminindex = min(tempdict, key=tempdict.get)
                            distance = tempdict[cminindex]
                            if distance < 4 * math.sqrt(dimension):
                                discard_cluster_points_dict[cminindex].append(point_index)
                                discardsetinfodict[cminindex]['N'] += 1
                                discardsetinfodict[cminindex]['SUM'] = \
                                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUM'],
                                                         point_coor)]
                                discardsetinfodict[cminindex]['SUMSQ'] = \
                                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUMSQ'],
                                                         [x ** 2 for x in point_coor])]
                                assigned = 1
                            if assigned == 0:
                                retainedsetdict[point_index] = point_coor
        for point_index, point_coor in ipmforassign.items():
            assigned = 0
            tempdict = {}
            for cluster_index, cluster_info in discardsetinfodict.items():
                distance = p2cdistma(point_coor, cluster_info)
                tempdict[cluster_index] = distance
            cminindex = min(tempdict, key=tempdict.get)
            distance = tempdict[cminindex]
            if distance < 4 * math.sqrt(dimension):
                discard_cluster_points_dict[cminindex].append(point_index)
                discardsetinfodict[cminindex]['N'] += 1
                discardsetinfodict[cminindex]['SUM'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUM'],
                                         index_point_map[point_index])]
                discardsetinfodict[cminindex]['SUMSQ'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUMSQ'],
                                         [x ** 2 for x in index_point_map[point_index]])]
                assigned = 1
            if assigned == 0:
                retainedsetdict[point_index] = point_coor

        intermediateheader = ['round_id', 'nof_cluster_discard', 'nof_point_discard',
                              'nof_cluster_compression', 'nof_point_compression',
                              'nof_point_retained']
        intermediateoutput.write(','.join(intermediateheader))
        intermediateoutput.write('\n')

    else:
        index_point_map = {d[0]: d[1:] for d in lines}
        for point_index, point_coor in index_point_map.items():
            assigned = 0
            tempdict = {}
            for cluster_index, cluster_info in discardsetinfodict.items():
                distance = p2cdistma(point_coor, cluster_info)
                tempdict[cluster_index] = distance
            cminindex = min(tempdict, key=tempdict.get)
            distance = tempdict[cminindex]
            if distance < 4 * math.sqrt(dimension):
                discard_cluster_points_dict[cminindex].append(point_index)
                discardsetinfodict[cminindex]['N'] += 1
                discardsetinfodict[cminindex]['SUM'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUM'],
                                         index_point_map[point_index])]
                discardsetinfodict[cminindex]['SUMSQ'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUMSQ'],
                                         [x ** 2 for x in index_point_map[point_index]])]
                assigned = 1

            if assigned == 0:
                retainedsetdict[point_index] = point_coor

    if indx == len(txt_paths):
        newretainedsetdict = {}
        for point_index, point_coor in retainedsetdict.items():
            assigned = 0
            tempdict = {}
            for cluster_index, cluster_info in discardsetinfodict.items():
                distance = p2cdistma(point_coor, cluster_info)
                tempdict[cluster_index] = distance
            cminindex = min(tempdict, key=tempdict.get)
            distance = tempdict[cminindex]
            if distance < 6 * math.sqrt(dimension):
                discard_cluster_points_dict[cminindex].append(point_index)
                discardsetinfodict[cminindex]['N'] += 1
                discardsetinfodict[cminindex]['SUM'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUM'],
                                         retainedsetdict[point_index])]
                discardsetinfodict[cminindex]['SUMSQ'] = \
                    [sum(i) for i in zip(discardsetinfodict[cminindex]['SUMSQ'],
                                         [x ** 2 for x in retainedsetdict[point_index]])]
                assigned = 1
            if assigned == 0:
                newretainedsetdict[point_index] = point_coor
        retainedsetdict = newretainedsetdict.copy()

    nof_cluster_discard = len(discardsetinfodict)
    nof_point_dicard = 0
    for key, val in discardsetinfodict.items():
        nof_point_dicard += discardsetinfodict[key]['N']
    nof_cluster_compression = len(compressedsetinfodict)
    nof_point_compression = 0
    for key, val in compressedsetinfodict.items():
        nof_point_compression += compressedsetinfodict[key]['N']
    nof_point_retained = len(retainedsetdict)
    intermediateoutput.write(str(indx) + ',' + str(nof_cluster_discard) + ',' + str(nof_point_dicard) + ','
                             + str(nof_cluster_compression) + ',' + str(nof_point_compression) + ','
                             + str(nof_point_retained) + '\n')
    indx += 1


id0 = 0
for cluster_id, points in discard_cluster_points_dict.items():
    newdict = {point: id0 for point in points}
    clusterdict.update(newdict)
    id0 += 1

outliers = sum(list(compressed_cluster_points_dict.values()), []) + list(retainedsetdict.keys())
newdict = {point: -1 for point in outliers}
clusterdict.update(newdict)

with open(out_file1, 'w') as f:
    json.dump(clusterdict, f)

print('Duration: ' + str(time.time() - start_time))
