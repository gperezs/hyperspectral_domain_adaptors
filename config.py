import numpy as np

dset_root = {}
dset_root['cub'] = '/scratch2/gperezsarabi/datasets/CUB_200_2011'
dset_root['cars'] = '/scratch2/gperezsarabi/datasets/CARS'
dset_root['aircrafts'] = '/scratch2/gperezsarabi/datasets/fgvc-aircraft-2013b'
dset_root['legus'] = '/scratch2/gperezsarabi/datasets/LEGUS'
dset_root['eurosat'] = '/scratch2/gperezsarabi/datasets/EuroSAT'
dset_root['so2sa'] = '/scratch2/gperezsarabi/datasets/So2Sa_LCZ42'

# means and stds 
means = {}
means['cub5']           = np.array([0.68739661, 0.60354770, 0.82736819, 0.77292223, 0.83835272])
means['cub15']          = np.array([0.76372112, 0.81395221, 0.78828949, 0.61539703, 0.83425286, 0.64343428,
                                    0.78010727, 0.74749959, 0.83511892, 0.69887701, 0.76897810, 0.49830110,
                                    0.81944520, 0.73204447, 0.84908764])
means['cars5']          = np.array([0.67893349, 0.57681889, 0.77480887, 0.71862066, 0.76684566])
means['cars15']         = np.array([0.69719173, 0.77029703, 0.74258469, 0.62377560, 0.76222643, 0.60972885,
                                    0.72830587, 0.69535610, 0.77148499, 0.65713616, 0.68826678, 0.49037044,
                                    0.75734247, 0.71180506, 0.78797339])
means['aircrafts5']     = np.array([0.64633522, 0.65052546, 0.80807712, 0.82163048, 0.84067126])
means['aircrafts15']    = np.array([0.77607808, 0.79652970, 0.82725096, 0.57219525, 0.83423034, 0.69118638,
                                    0.73661759, 0.79109880, 0.81276638, 0.76827406, 0.73186304, 0.54437124,
                                    0.85216642, 0.69441736, 0.85569445])
means['legus']          = np.array([0.02602760, 0.04746260, 0.15163783, 0.29678162, 0.37417257])
means['eurosat']        = np.array([5.30873304, 4.38118546, 4.08582245, 3.71197747, 4.70270065, 7.85492863,
                                    9.30983704, 9.02439388, 2.87130177, 0.04744913, 7.13998579, 4.38510872,
                                    10.1952272])
means['so2sa']          = np.array([-3.59122426e-05, -7.65856128e-06,  5.93738575e-05,  2.51662315e-05,
                                     4.42011066e-02,  2.57610271e-01,  7.55674337e-04,  1.35034668e-03,
                                     1.23756961e-01,  1.09277464e-01,  1.01085520e-01,  1.14239862e-01,
                                     1.59265669e-01,  1.81472360e-01,  1.74574031e-01,  1.95016073e-01,
                                     1.54284689e-01,  1.09050507e-01])

stds = {}
stds['cub5']            = np.array([0.04980726, 0.05031700, 0.03465287, 0.03821204, 0.02760218])
stds['cub15']           = np.array([0.03864040, 0.03747835, 0.02490082, 0.05210588, 0.02783384, 0.04894743,
                                    0.03963811, 0.04246485, 0.02993877, 0.03834033, 0.02605400, 0.05196682,
                                    0.03196410, 0.04731990, 0.02620091])
stds['cars5']           = np.array([0.04756925, 0.04751240, 0.03513504, 0.03629709, 0.02424341])
stds['cars15']          = np.array([0.03551516, 0.03836373, 0.02449360, 0.04926482, 0.02432199, 0.04607141,
                                    0.03868891, 0.04016711, 0.02906602, 0.03672580, 0.02122876, 0.05037721,
                                    0.02962270, 0.04593521, 0.02499772])
stds['aircrafts5']      = np.array([0.03014784, 0.02950053, 0.02644797, 0.02540198, 0.01762077])
stds['aircrafts15']     = np.array([0.02386056, 0.02835769, 0.01961588, 0.03073552, 0.01756325, 0.02888254,
                                    0.02687966, 0.02683579, 0.02185843, 0.02569087, 0.01472071, 0.03125018,
                                    0.02156265, 0.02985248, 0.01925699])
stds['legus']           = np.array([0.03456852, 3.69824263, 0.13778579, 0.20603000, 0.24007532])
stds['eurosat']         = np.array([0.07618699, 0.18407695, 0.21485147, 0.30419708, 0.25214213, 0.35755264,
                                    0.44993155, 0.46499325, 0.10017679, 0.00104862, 0.39385335, 0.32988129,
                                    0.48987974])
stds['so2sa']           = np.array([0.04205938, 0.04202206, 0.12461424, 0.12439287, 0.08873851, 0.7959068,
                                    0.12153513, 0.07501314, 0.00495793, 0.00563989, 0.00712047, 0.0061751,
                                    0.00773704, 0.00897009, 0.00960729, 0.00977149, 0.00845526, 0.00756963])


# kmeans centers for 5 and 15 channels (for synthetic datasets)
centers = {}
centers['5']  = np.array([[ 45.02998717,  46.60168392,  30.19152471],
                          [220.25319838, 221.47942769, 215.55068774],
                          [ 96.39331631, 101.46956127,  74.95141297],
                          [150.85427457, 168.15904154, 182.94842101],
                          [149.87229554, 146.47161387, 109.39811333]])
centers['15'] = np.array([[193.25607576, 185.11730208, 130.32975473],
                          [ 86.00396873,  87.19882075,  77.68599076],
                          [ 73.86705090, 116.11337091, 158.98960406],
                          [ 24.81336652,  25.86829491,  17.14159180],
                          [155.85595489, 146.20887970, 106.72095403],
                          [208.73840671, 210.76156662, 207.94210920],
                          [ 88.55250859, 109.26432213,  33.04006676],
                          [177.50408457, 180.61637386, 181.31557229],
                          [123.75189793, 114.42222468,  72.20789786],
                          [124.98632585, 173.41673657, 223.54308964],
                          [158.06481034, 166.67107012,  45.52685228],
                          [243.72321467, 245.29483684, 242.78711426],
                          [144.03924618, 151.19731132, 154.09411315],
                          [ 59.53954279,  59.25566351,  40.69977230],
                          [115.94298950, 119.79448229, 113.42798463]])

# CUB (100):
centers['cub5']  = np.array([[ 91.68377663,  98.36193761,  61.28592175],
			     [157.69826356, 166.96125510, 169.90795988],
			     [140.06402019, 138.21925966, 106.10118722],
 			     [ 39.24136080,  42.11136882,  26.09451966],
			     [214.75004594, 220.71132946, 220.50884215]])

centers['cub15'] = np.array([[131.99181864, 135.89191157, 134.93808169],
			     [206.83809394, 213.55633636, 223.18065569],
			     [ 45.64154822,  48.39789824,  34.24922735],
			     [105.90460655, 108.31549001,  98.25452536],
			     [181.42849313, 182.99246959,  74.29349604],
			     [134.84093324, 167.76844342, 205.44571537],
			     [ 74.52887474,  82.91124905,  25.56322516],
			     [173.76739652, 165.59074460, 144.72548608],
			     [ 15.57038685,  16.57995674,  12.80512678],
			     [195.37774391, 193.99723452, 182.07034767],
			     [242.06150835, 244.60058355, 244.39234877],
			     [139.80906110, 139.78374386,  93.74353539],
			     [144.79712372, 110.43251615,  47.73170802],
   			     [ 70.03917838,  77.92961983,  71.48482853],
		 	     [ 97.41486185, 111.25578045,  55.49639609]])


