import sys
from statistics import mode

training_data_2c = [[0.05, 0.037470225880689, 0.14722887544019, 6.2805083450785, "Ceramic"],
                    [0.12126354869863, 0.07931037144998,
                        0.38672181315129, 3.3679640192997, "Metal"],
                    [0.13422941031488, 0.053361555885849,
                        0.17353797970858, 2.9565915756204, "Plastic"],
                    [0.16173100074975, 0.15, 0.55739218346738,
                        3.5629545824234, "Plastic"],
                    [0.11140621197346, 0.076688863457611,
                     0.21180536029907, 3.8881451745148, "Plastic"],
                    [0.10390290019584, 0.072630796554611,
                     0.35093430155169, 2.5215027718218, "Metal"],
                    [0.063842616019911, 0.068497865994088,
                     0.15635833391939, 2.9999944977949, "Metal"],
                    [0.080103483661813, 0.073627428984742,
                     0.30246343031383, 5.3963285382531, "Metal"],
                    [0.081633407403848, 0.066983148821293,
                     0.26489162099776, 0.96974134713235, "Metal"],
                    [0.10320463844187, 0.09559480177698,
                     0.65821574273178, 3.5112235050873, "Ceramic"],
                    [0.061929454852413, 0.03, 0.1, 5.0629067815504, "Metal"],
                    [0.061407521523095, 0.05587429775472,
                     0.17024109445144, 2.7172831405484, "Metal"],
                    [0.073726306626722, 0.03, 0.14861433062848,
                        2.4979621069546, "Ceramic"],
                    [0.21657041224651, 0.15, 0.75, 3.0544311261072, "Plastic"],
                    [0.12097311317698, 0.090155763788295,
                     0.69967541295806, 3.817620761528, "Ceramic"],
                    [0.12162397059681, 0.104206871201,
                        0.53075294439767, 1.6795182762687, "Metal"],
                    [0.088221709397026, 0.079170757380562,
                     0.32016258103101, 2.8848727947135, "Metal"],
                    [0.097234766570623, 0.071694981531227,
                     0.43408157579128, 2.6677427857872, "Ceramic"],
                    [0.085637989275061, 0.060869481199544,
                     0.25977187587144, 2.2651944335794, "Metal"],
                    [0.14139819475477, 0.099260965107592,
                        0.75, 4.4299943834668, "Ceramic"],
                    [0.10346283035796, 0.095692295221534,
                     0.58622577186891, 2.2192479686997, "Ceramic"],
                    [0.11049327587641, 0.095585187925297,
                     0.66262385846408, 3.5727993179765, "Ceramic"],
                    [0.12344433723453, 0.11387548440696,
                        0.58114988746542, 3.1900559390537, "Metal"],
                    [0.16813556499321, 0.13242793138042,
                     0.49987254442095, 3.1063195534242, "Plastic"],
                    [0.15536954509006, 0.13524553345437,
                        0.75, 3.9939633950696, "Ceramic"],
                    [0.14789243491391, 0.15, 0.51709733940442,
                        2.8177090177733, "Plastic"],
                    [0.16737426855801, 0.13829857485537,
                     0.60460259544045, 2.7827999384897, "Plastic"],
                    [0.052470411265718, 0.047879298506296,
                     0.12719055769244, 4.6749640575688, "Metal"],
                    [0.13317106159108, 0.11429472643311,
                     0.35520000516667, 3.3148490032695, "Plastic"],
                    [0.12154724009815, 0.1379144557241,
                        0.75, 1.750619810353, "Ceramic"],
                    [0.12063221388773, 0.11638999479219,
                        0.59064761191289, 2.8303553708397, "Metal"],
                    [0.096929818339593, 0.080571368664893,
                     0.3382926904285, 1.6804677964357, "Metal"],
                    [0.084194585661758, 0.074644478327638,
                     0.2934745021292, 2.4462266854975, "Metal"],
                    [0.061597582551352, 0.03, 0.12027378772253,
                        1.6370388076607, "Metal"],
                    [0.07593413381105, 0.058196801605464,
                        0.22712802572997, 2.380087163498, "Metal"],
                    [0.10537385863014, 0.10427470292995,
                     0.73281356010129, 2.6639932840659, "Ceramic"],
                    [0.11468799739486, 0.1015768227512,
                        0.49348143397665, 2.5330717316676, "Metal"],
                    [0.16747934324625, 0.094530271578422,
                     0.36079808175778, 3.5824523203869, "Plastic"],
                    [0.1299156781363, 0.1490409361, 0.49441847769997,
                        0.5806443308568, "Plastic"],
                    [0.16576926558299, 0.15, 0.55948675625836,
                        3.9309005920167, "Plastic"],
                    [0.077862771929204, 0.060039463699145,
                     0.19368244224634, 1.3819709246581, "Metal"],
                    [0.15059711326561, 0.15, 0.53341450336939,
                        1.7483447105026, "Plastic"],
                    [0.18577410353864, 0.15, 0.57937987748382,
                        4.3352848172381, "Plastic"],
                    [0.084598602002561, 0.044527964458387,
                     0.15748670319746, 3.661976914976, "Metal"],
                    [0.10719891223113, 0.083666019991313,
                     0.23390982200268, 3.0946672431349, "Plastic"],
                    [0.16562689508245, 0.15, 0.58042950248961,
                        3.1216127011465, "Plastic"],
                    [0.17700413560341, 0.15, 0.61136723449668,
                        4.6747065110148, "Plastic"],
                    [0.080450685210163, 0.067155430460717,
                     0.2712457754559, 4.7608135627083, "Metal"],
                    [0.091970281060154, 0.048449692673522,
                     0.16903463884971, 4.0522987027037, "Metal"],
                    [0.12216524919887, 0.15, 0.42980201416389,
                        3.7513971983461, "Plastic"],
                    [0.067309129709123, 0.06599301306974,
                     0.19697098066782, 1.7739365546538, "Metal"],
                    [0.081372163889279, 0.11109365739604,
                     0.55321227688217, 2.0614483699517, "Ceramic"],
                    [0.13986609908878, 0.089150039247582,
                     0.29650575649855, 2.709527261076, "Plastic"],
                    [0.074712222098086, 0.030104290261049,
                     0.18368789719972, 3.527968322563, "Ceramic"],
                    [0.13630114295041, 0.096697923947344,
                     0.32567682970765, 4.1712705127324, "Plastic"],
                    [0.11077798339855, 0.040308171028745,
                     0.32013975306574, 3.2811819346404, "Ceramic"],
                    [0.06738763377091, 0.059906344708468,
                     0.2977656676607, 3.6905142631241, "Ceramic"],
                    [0.072963551174142, 0.03423578426205,
                     0.12433596879049, 3.8603879023673, "Metal"],
                    [0.15000543910928, 0.12611138456117,
                     0.43393775904175, 3.1681506274315, "Plastic"],
                    [0.073782229913215, 0.073994036410239,
                     0.2701168135206, 4.5027567371314, "Metal"],
                    [0.074349798506463, 0.069803487750464,
                     0.23493940416864, 2.0562867922551, "Meta"],
                    [0.084346840237618, 0.076669757354978,
                     0.28963829889643, 4.306370223048, "Metal"],
                    [0.095631099272473, 0.10897713540401,
                     0.63618732311381, 3.9224639243484, "Ceramic"],
                    [0.14778992863496, 0.13540843455298,
                        0.75, 4.4231696303781, "Metal"],
                    [0.074941107239757, 0.076881363519226,
                     0.24283468217744, 2.010919319306, "Metal"],
                    [0.10215861135343, 0.14140796228293,
                        0.75, 3.6526364386602, "Ceramic"],
                    [0.148436568769, 0.15, 0.49802858147598,
                        4.805754113822, "Plastic"],
                    [0.095208661651332, 0.10304783355452,
                     0.42076269691515, 5.3952652576931, "Metal"],
                    [0.11619772849505, 0.082442704940994,
                     0.27848032936225, 4.10367414877, "Plastic"],
                    [0.12415834855693, 0.11680059511522,
                        0.75, 4.4032624366325, "Ceramic"],
                    [0.11072175387353, 0.12680094635068,
                        0.36323548926785, 3.944332283933, "Plastic"],
                    [0.15286612039629, 0.11430430498828,
                     0.38203186483226, 3.4655211053227, "Plastic"],
                    [0.092635803569435, 0.087748580942444,
                     0.55549812788935, 3.5137129357662, "Ceramic"],
                    [0.12909751741303, 0.074568832077216,
                     0.63743215089559, 2.0343838221142, "Ceramic"],
                    [0.058830601438825, 0.086005662371459,
                     0.30771603089303, 3.3649439773798, "Ceramic"],
                    [0.13490226463673, 0.11731094255821,
                        0.75, 2.5348140081938, "Ceramic"],
                    [0.12001227069482, 0.1158209201071,
                        0.75, 2.3665721775781, "Ceramic"],
                    [0.089693973592511, 0.06198108998296,
                     0.38361497179421, 4.0533258630768, "Ceramic"],
                    [0.17907702930991, 0.15, 0.64506317656166,
                        2.8591689134771, "Plastic"],
                    [0.12954648314476, 0.131861044643,
                        0.36954335557186, 2.9177595690373, "Plastic"],
                    [0.19466087501022, 0.15, 0.61538248015557,
                        2.9561130070774, "Plastic"],
                    [0.16340124463654, 0.14379068708953,
                     0.48365522205498, 3.1014109431797, "Plastic"],
                    [0.10535318168686, 0.12837236987807,
                        0.75, 3.4384517673824, "Ceramic"],
                    [0.093296748738269, 0.039956280156703,
                     0.23828815404119, 2.5109696789376, "Ceramic"],
                    [0.15129258309659, 0.10062787971303,
                     0.38303471067222, 3.1661184772817, "Plastic"],
                    [0.11919549496429, 0.085091876662302,
                     0.70239884202334, 2.9074682491548, "Ceramic"],
                    [0.090999458732459, 0.094177740559965,
                     0.53174653467802, 0.87404303798773, "Ceramic"],
                    [0.097981098790882, 0.087596477988012,
                     0.49204504851025, 2.2039768489419, "Ceramic"],
                    [0.10218716578535, 0.07966612719785,
                     0.57466012337118, 4.2396845403335, "Ceramic"],
                    [0.13292117858255, 0.12112772092469,
                     0.39980986091612, 3.3767187220826, "Plastic"],
                    [0.12899521780698, 0.10499569860475,
                     0.25642981914901, 2.8587411246251, "Plastic"],
                    [0.13588735558731, 0.091701585317159,
                     0.33746891179469, 3.369515328686, "Plastic"],
                    [0.10801784050922, 0.12684292883576,
                     0.24413988409528, 3.1337458076828, "Plastic"],
                    [0.14031389121844, 0.13149470493601,
                     0.39294675928757, 2.1263485078811, "Plastic"],
                    [0.13786420327606, 0.054809266594206,
                     0.19892671878224, 4.0131390233802, "Plastic"],
                    [0.086373978373777, 0.03, 0.11919499699437,
                        4.190669488204, "Plastic"],
                    [0.16833106316401, 0.15, 0.53482409379436,
                        5.1268259676957, "Plastic"],
                    [0.13499515462452, 0.13096283058287,
                        0.3707348603947, 3.5026195312603, "Plastic"],
                    [0.19635256219694, 0.15, 0.59496548338726,
                        3.1484400263874, "Plastic"],
                    [0.18675839570366, 0.15, 0.60869504932821,
                        3.2978034171522, "Plastic"],
                    [0.15799751295221, 0.15, 0.4935777059827,
                        4.5742250832736, "Plastic"],
                    [0.16033879353829, 0.15, 0.50008707229488,
                        1.8847205013657, "Plastic"],
                    [0.13637285015034, 0.10069888886748,
                     0.28395725229764, 4.2355856982484, "Plastic"],
                    [0.15252654467777, 0.13062456939749,
                        0.4479681641844, 1.5393977761263, "Plastic"],
                    [0.14551391447064, 0.1177224567855,
                        0.43532078724964, 4.1881289583028, "Plastic"],
                    [0.1718759672599, 0.15, 0.55608986711122,
                        3.830577904698, "Plastic"],
                    [0.10509321664011, 0.090266578940531,
                     0.18829867889817, 3.4818951297628, "Plastic"],
                    [0.14133343228316, 0.13693864634062,
                        0.51315221769061, 3.368829789043, "Plastic"],
                    [0.15457258405086, 0.15, 0.59346882385712,
                        3.0068476767674, "Plastic"],
                    [0.15108760338215, 0.15, 0.54259915482501,
                        3.7826898611092, "Plastic"],
                    [0.17472946378896, 0.15, 0.55066427391698,
                        2.7753396365925, "Plastic"],
                    [0.15541951086881, 0.14016639224233,
                     0.44581822765113, 2.0641051222407, "Plastic"],
                    [0.16590797720015, 0.15, 0.52452213489581,
                        5.478681845518, "Plastic"],
                    [0.18161503340507, 0.15, 0.55275102984289,
                        1.8570200826993, "Plastic"],
                    [0.15725979273113, 0.14971610831672,
                     0.54369960775049, 3.9432246826883, "Plastic"],
                    [0.16191252766821, 0.15, 0.51250770134215,
                        4.7795695104584, "Plastic"],
                    [0.19507030481136, 0.15, 0.60951036566472,
                        2.6996518465231, "Plastic"],
                    [0.1721261183648, 0.15, 0.52176917247779,
                        2.766020167521, "Plastic"],
                    [0.11386317464533, 0.14847819267259,
                     0.38417090635405, 3.5023069764584, "Plastic"],
                    [0.14755873238988, 0.14147217321605, 0.40111900265645, 2.9630039749106, "Plastic"]]

training_data_2a = [[0.12314928471509, 0.11343071085638, 0.59238897131847, 4.391234005624, "Metal"],
                    [0.11824043312801, 0.072647143934747,
                        0.52935375120921, 4.3621982231196, "Ceramic"],
                    [0.057884429333034, 0.066632797285605,
                        0.15236920541698, 4.7063921119129, "Metal"],
                    [0.11286970369839, 0.047061063896848,
                        0.1, 2.5728878957994, "Plastic"],
                    [0.18274001395343, 0.14544182984989,
                        0.6720262155531, 1.968789385143, "Plastic"],
                    [0.10704958910424, 0.10642928903432,
                        0.50349892909243, 2.5826989277375, "Metal"],
                    [0.067106026012441, 0.11465560797346,
                     0.48276125021255, 3.9247493377322, "Ceramic"],
                    [0.1407873350872, 0.11275731177491,
                        0.75, 2.1315773470431, "Ceramic"],
                    [0.12747909172499, 0.11124419685207,
                     0.34597151759956, 4.4037809292331, "Plastic"],
                    [0.14413769755615, 0.15, 0.47834110729103,
                        3.5510026384028, "Plastic"],
                    [0.16893960792298, 0.15, 0.54038114380453,
                        1.4286629197489, "Plastic"],
                    [0.19865682546668, 0.15, 0.66884634820396, 3.4418559131262, "Plastic"]]


test_data_2a = [[0.088631206737882, 0.068539262673328, 0.2613098023888, 4.8865293207087],
                [0.11370698832306, 0.082043720295394,
                    0.40384077886289, 3.5782314354847],
                [0.11826703276113, 0.051432342913349,
                    0.35467621587675, 3.0855576890382],
                [0.1799528787649, 0.15, 0.55451001761916, 2.2370858537624]]

# calculates_cartesian_distance


def cal_cartesian_distance(train_X, new_test_X, que):
    distances = []
    count = 1
    for j in train_X:
        eucledian_distance = 0
        for i in range(len(new_test_X)):
            eucledian_distance = eucledian_distance+(new_test_X[i]-j[i])**2
        cartesian_distance = (eucledian_distance)**0.5
        distances.append(cartesian_distance)
        if que in ["2a", "2b", "2d", "2e"]:
            print(count, ". cartesian_distance between ",
                  new_test_X, "and ", j, "is ", cartesian_distance)
            count += 1
    return distances

# calulates manhatten_distance


def cal_manhatten_distance(train_X, new_test_X, que):
    distances = []
    count = 1
    for j in train_X:
        base_distance = 0
        for i in range(len(new_test_X)):
            base_distance = base_distance+abs(new_test_X[i]-j[i])
            if que in ["2a", "2b", "2d", "2e"]:
                print(count, ". manhatten_distance between ",
                      new_test_X, " and ", j, "is ", base_distance)
        distances.append(base_distance)
        count += 1
    return distances


# find the closest neighbour
def find_closest_neighbor(new_test_X, K, train_X, similarity, que):
    closest_neigbours = []
    distances = []
    if similarity == "cartesian":
        distances = cal_cartesian_distance(train_X, new_test_X, que)
    else:
        distances = cal_manhatten_distance(train_X, new_test_X, que)
    sorted_distances = distances.copy()
    sorted_distances.sort()
    dic = {}
    dic_loc = 0
    # hashmap to find the point using distance in constant time
    for i in distances:
        dic[i] = dic_loc
        dic_loc += 1
    for i in range(K):
        closest_neigbours.append(dic[sorted_distances[i]])
    return closest_neigbours

# maps the nearest neighbor point to its target


def get_classes(i, train_target):
    class_list = []
    for j in i:
        class_list.append(train_target[j])
    return class_list

# prediction function for 2c 2d 2e


def predict(K, test_data, new_train_x, similarity, train_target, que):
    neighbor = find_closest_neighbor(
        test_data, K, new_train_x, similarity, que)
    classes = get_classes(neighbor, train_target)
    if que in ["2d", "2e"]:
        print("----------------------------------------------------------------", "the closest neighbors of ", test_data, " are ", neighbor,
              "there classes are ", classes, "so these are the selected neighbors")
        print("The prediction is ", mode(classes),
              "------------------------------------------------------------------------\n")
    return mode(classes)

# prediction function for both 2a and 2b


def predict_2a(K, test_data, new_train_x, train_target, similarity, que):
    for i in test_data:
        neighbor = find_closest_neighbor(i, K, new_train_x, similarity, que)
        classes = get_classes(neighbor, train_target)
        print("the closest neighbors of ", i, " are ", neighbor,
              "there classes are ", classes, "so these are the selected neighbors")
        print("The prediction is ", mode(classes))
    return mode(classes)

# leave one routine implementation


def leave_one_routine(K, train_X, train_target, similarity, que):
    count = 0
    for i in range(len(train_X)):
        # copy of train is made to avoid modification of actual trainx as it can effect the next iteractions due to py references
        new_train_x = train_X.copy()
        new_test = new_train_x.pop(i)
        new_test_target = train_target[i]
        prediction = predict(K, new_test, new_train_x,
                             similarity, train_target, que)
        if prediction == new_test_target:
            count = count+1
    # count variable storess the number of correct predictions and using that accuracy is calculated
    print("accuracy of the prediction given k= ", K, ",similarity measure=",
          similarity, ",diminsions,", len(train_X[0]), "is", count/len(train_X))
    return count

# code classifies which question it should solve based on the input


def driver(train_X, train_target, K, test_data, que):
    match que:
        case "2a": return predict_2a(K, test_data, train_X, train_target, "cartesian", que)
        case "2b":
            test_data = [[0.12126354869863, 0.07931037144998,
                          0.38672181315129, 3.3679640192997]]
            return predict_2a(K, test_data, train_X, train_target, "cartesian", que)
        case "2c":
            x = leave_one_routine(K, train_X, train_target, "cartesian", que)
            return x/len(train_X)
        case "2d":
            x = leave_one_routine(K, train_X, train_target, "manhatten", que)
            return x/len(train_X)
        case "2e":
            x = leave_one_routine(K, train_X, train_target, "cartesian", que)
            return x/len(train_X)

# selectes the data that needs to be feeded for the specific question and preprocess that accordingly


def preprocessing(training_data_2a, training_data_2c, que):
    train_X = []
    train_target = []
    if que in ["2a", "2b"]:
        for i in training_data_2a:
            train_X.append(i[:-1])
            train_target.append(i[-1])
        return train_X, train_target
    elif que == "2e":
        for i in training_data_2c:
            train_X.append(i[:-2])
            train_target.append(i[-1])
        return train_X, train_target
    else:
        for i in training_data_2c:
            train_X.append(i[:-1])
            train_target.append(i[-1])
        return train_X, train_target

# runner function preprosses the data as required for the specific question and calls the driver function


def runner(training_data_2a, test_data_2a, training_data_2c, que):
    train_X, train_target = preprocessing(
        training_data_2a, training_data_2c, que)
    print("output for ", que,
          "#########################################################################################################")
    for K in range(1, 7, 2):
        print("current k value= ", K,
              "*******************************************************************************")
        driver(train_X, train_target, K, test_data_2a, que)


# testing file is created and output of all questions is stored in it
f = open("testing.txt", 'w')
sys.stdout = f
for i in ["2a", "2b", "2c", "2d", "2e"]:
    runner(training_data_2a, test_data_2a, training_data_2c, i)
f.close()

# uncomment and run the code to output the results into textfile called testing.txt
# testing file is created and by changing question number in runner but have to run manually for each question
# f = open("testing.txt", 'w')
# sys.stdout = f
# runner(training_data_2a, test_data_2a, training_data_2c, "2e")
# f.close()

# To see the output inside terminal , one after the other uncomment below line
# runner(training_data_2a, test_data_2a, training_data_2c, "2e")


'''
observations:
k   acc   fetaurecount   simmilarity 
1   0.55       4            cart
3   0.575      4            cart
5   0.55       4            cart
1   0.733      3            cart
3   0.758      3            cart
5   0.775      3            cart
1   0.6        4            man
3   0.633      4            man
5   0.6        4            man
1   0.7416     3            man
3   0.766      3            man
5   0.808      3            man
'''
