import numpy as np
from scipy.signal import butter, lfilter, freqz,filtfilt
import matplotlib.pyplot as plt
mu, sigma = 0,500


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5, plots=True):
    ''' fs : sampling freq. (pts/s) '''
    cutoff = float(cutoff)
    fs = float(fs)
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def main():
    #plt.plot(x,y,linewidth=2,linestyle="-", c="b") #it includes some noise

    n= 33
    b = [0.2/n]*n
    a = 1

    #z = np.random.normal(mu, sigma, len(x))#noise
    #y = x**2 + z #data

    #yy = butter(a,y,btype='lowpass',analog=False)
    #yy = lfilter(b,a,y)
    y = [0,27.00964394447541,1940.4409132697688,529.5090440681838,46.19995416523209,222.61056703431652,15.375249610936292,114.58003412298332,174.4775726873386,184.70096289487353,42.63476660743744,47.499811173760776,98.0785058819356,72.69208556283374,49.38767793727101,278.8684800579696,269.46257290677795,263.2473096363623,57.59565874648988,207.5583490756077,118.0658514375354,798.8357541028068,238.53481982965525,220.2428763723186,135.64131192413072,172.35263110605487,44.19952313620941,275.04666982381184,290.985013665045,154.60912995649238,166.972215834916,53.15384263626238,296.420261204611,88.06884349678255,17.47439598297435,61.97197436356563,217.4648460737123,77.12291961371531,430.72077717722334,335.6292762358056,259.2671796282205,430.993691069882,84.51933529991342,119.7125550336699,130.07755931897302,154.20143581661844,209.83696634167714,335.43590062917843,15.379313296398344,579.7451353986369,5.71415117075898,49.61674758827764,323.4920134941115,368.27001991400454,129.80946569282028,345.9032074183923,45.6582487985888,133.22187576200318,81.47010997575335,597.9349218656657,214.26376914486076,286.0671093368907,212.81434618239564,141.25851748139803,185.60089183580675,40.23257520294708,13.098983366397919,670.8723632983993,1039.947545145035,272.96236373182023,279.50602698505594,855.7096781115662,340.5211739002041,427.7466476468428,437.87986845218745,230.67228530004286,145.12822920737304,464.84378314587576,178.574934585881,645.1193944652582,750.4257873669252,75.3768619817627,556.3264316113706,224.57599218923136,214.92420577005566,55.50459499365109,183.29164203518826,97.34865008245536,171.84466712165857,93.31245764608501,307.03906502657134,30.41898950871037,183.22717888832375,214.7311842445262,92.92386292283031,3.7477168946218495,365.76200052846104,320.91305888828407,224.39538107070408,106.45868804383196,16.530428643641653,244.1483783512055,73.59876809975508,394.96690503756133,362.49255308324314,668.4444597254396,214.40165551347388,108.87910978252646,184.38322882496402,103.49984640865199,86.572607901814,441.6539320747399,360.07376947593036,107.0858401961371,152.41260131752503,145.66764624461953,262.42248183558627,39.45066819150221,254.03169760436268,272.5990757822714,128.32319673618784,148.21917222406304,342.41421874630197,525.8868745548733,41.38044666379901,108.87331833310324,511.57472367212114,436.84707851827517,59.514817880855645,304.7289747160995,172.25733181251238,111.12012197108345,36.33484496468567,514.6895112080974,541.3020675849159,99.37298747127129,302.65669358071744,172.45659673756919,348.234159890482,80.32219250308577,182.91999581612174,66.6331962120391,340.1432620283067,183.99740079623928,119.39952437434411,226.8644412172263,268.10470614653786,456.6659226797078,159.28010714086395,201.52548969930797,135.21376402720938,476.0263371491016,474.97851875950016,157.63453901993978,274.912257507077,41.250886627292715,296.5584587177045,98.7475481660569,283.39422133012766,179.66862558691247,77.81816475114242,94.49125835895772,427.19070304616145,290.769681413239,178.25990345913067,28.850436989693126,69.84551826766047,214.233111296217,16.0812889299954,246.97505599108624,154.62529055085372,99.28094500896378,76.35551393529221,148.35209539017654,31.178771399302974,52.55311117397025,75.70747868428082,64.86578329883379,179.3360069697297,188.74738200755561,19.204525377752983,177.30461937010608,177.2909849581953,11.986474551425031,0.0,15.676607587325972,35.67188827084679,24.479144786790307,57.57712519405134,70.62056926581197,67.90917899660353,35.24273184172456,37.944559541396266,30.34981352225851,25.198700523995743,7.231272870615086,54.16510961791104,85.42923754028766,38.04919526338912,34.457706775257456,19.1717963711812,42.25117520928339,0.0,84.2323017161529,0.0,4.659847216595675,75.70425585516821,54.483396451325056,28.78451415059229,110.29737674159932,216.13023168844717,139.23499362144327,45.98121474125463,52.17035022260318,22.495298285712824,122.22948073452648,29.73691772890235,102.40992073281757,0.0,13.585039476581704,34.392129237519455,61.38299235085158,18.485105378085578,48.17782205246329,60.562064073972444,91.03832020377548,152.42227785996633,32.343072394032866,0.0,0.0,46.082233107386045,0.0,0.0,48.56369552717728,7.231272870615086,35.53540059565255,26.32799483517341,75.28031519206004,21.364362669375513,8.569957639844786,34.68972439409637,17.62024068836711,20.13184724689319,28.914061239284248,25.86077584454641,0.0,20.47355306259782,12.781468427689193,6.46878735360501,8.49021111496084,51.88888976790336,44.16310325005231,23.398180338063483,0.0,53.23181365622467,0.0,135.2723656878047,0.0,112.92185329303572,9.055829400414973,23.72107491813109,0.0,4.449896766187796,24.568068436714803,46.69786495913605,0.0,30.35252678321493,168.03887881593232,50.45957535755143,43.59101614965081,31.569905214136462,135.8202199151088,245.0187329885969,152.4247544497264,268.690700624664,186.3618744131966,115.4724311836566,143.50688924126862,296.3306482901768,57.162087201559565,366.6665823635491,226.15054365460503,252.58180997207955,192.31316516079403,281.82311139126,92.43323233779078,841.4608479699433,319.1774039491264,62.822849427761845,92.00120331522805,456.50254445969614,19.874344332171166,371.3622098695793,44.119863449171326,377.86801497660514,15.3328868694476,152.32149557207276,17.270153921053325,778.284991597923,601.2335797146785,218.0810520563359,160.25081784249363,357.5875954825397,354.25373782864796,68.65226098348474,327.77576240999605,259.3884180971345,675.891338649014,88.28791194548238,212.56901868561985,187.77814708427204,79.71669793015472,74.09676367457784,571.4347004653079,875.7027468973861,78.75185652124392,524.6084961599064,223.14158029975798,185.96040533451185,108.39031142750254,849.4755245721093,526.7323388036331,311.8699043021662,242.48289699908136,95.23297276183855,312.4388478116335,31.854499826550327,374.38834484486694,107.48821059741393,520.2156950307842,11.91826810538181,469.49913226199124,695.3208680694601,78.809119455369,308.2774527253448,226.61825892285762,492.0062485114559,47.187795712573944,566.4406457070115,299.650754446921,231.69262752431177,440.69965403487146,88.20182133954222,503.8656289950938,211.67679041273647,74.56728484219853,103.25993763783823,170.24790439987578,221.24340649093966,2.0440382210857386,528.1976586636463,460.12941071321023,24.45764656358508,284.43022021088586,323.9362245893251,17.402440190294946,85.36974958922153,196.8790184042522,68.77939293018335,194.4733891029075,27.871836752183626,151.9355129164133,72.41816274963611,71.02153791336845,24.227266337585384,237.62153843312342,51.63995219674573,48.92446676811857,24.73209943275094,3.1499688421792498,116.63885639391363,163.647755659671,13.582772730758998,101.24866950446464,7.1477549562268345,3.6303535046169286,47.11394088039748,43.734198391295436,46.27657835697076,65.86236113280619,24.003980932048965,91.75663031199056,52.706925806688325,56.18409884257179,62.99306271654857,27.423581388573645,15.930534968814513,72.29884888953158,60.57700492926361,27.528846294010286,35.53407691984998,7.9668625045423775,48.66710564143463,2.4429837367468656,87.28984805541793,37.0523533781504,47.66695053225655,154.750758503499,609.3993076823224,202.5823289267752,734.7331500543524,17.20760435416333,590.6716754389746]

    x = np.arange(0,y.__len__(),1)
    yy = lowpass_filter(y, 0.1, 1, order=2)
    yy[:] = [x / 10 for x in yy]
    #yy = lfilter(b,a,y)
    #yy = savgol_filter(y, 20, 2, mode='nearest')


    #x axis
    plt.plot(x,yy,linewidth=2,linestyle="-", c="b") #it includes some noise
    #plt.show()
    #plt.plot(x,yy,linewidth=2,linestyle="-", c="b") #it includes some noise
    plt.show()

if __name__ == "__main__":
    main()