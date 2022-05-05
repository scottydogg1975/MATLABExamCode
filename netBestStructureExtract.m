%add word encodings to netbest structure
netBest.encSpanish = encSpanish;
netBest.encEnglish = encEnglish;

D = datetime("now",Format="yyyy_MM_dd__HH_mm_ss");
filename = "net_best__" + string(D) + ".mat";
save(filename,"netBest");

%extract best
netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;