import graphics
import data
import models

import pylab as pl

countries = 'ABW,AFG,AGO,ALB,ANT,ARE,ARG,ARM,AUS,AUT,AZE,BDI,BEL,BEN,BFA,BGD,BGR,BHR,BHS,BIH,BLR,BLZ,BOL,BRA,BRB,BRN,BTN,BWA,CAF,CAN,CHE,CHL,CHN,CIV,CMR,COD,COG,COL,COM,CPV,CRI,CUB,CYP,CZE,DEU,DJI,DNK,DOM,DZA,ECU,EGY,ERI,ESH,ESP,EST,ETH,FIN,FJI,FRA,FSM,GAB,GBR,GEO,GHA,GIN,GLP,GMB,GNB,GNQ,GRC,GRD,GTM,GUF,GUM,GUY,HKG,HND,HRV,HTI,HUN,IDN,IND,IRL,IRN,IRQ,ISL,ISR,ITA,JAM,JOR,JPN,KAZ,KEN,KGZ,KHM,KOR,KWT,LAO,LBN,LBR,LBY,LCA,LKA,LSO,LTU,LUX,LVA,MAC,MAR,MDA,MDG,MDV,MEX,MKD,MLI,MLT,MMR,MNE,MNG,MOZ,MRT,MTQ,MUS,MWI,MYS,NAM,NCL,NER,NGA,NIC,NLD,NOR,NPL,NZL,OMN,PAK,PAN,PER,PHL,PNG,POL,PRI,PRK,PRT,PRY,PSE,PYF,QAT,REU,ROU,RUS,RWA,SAU,SDN,SEN,SGP,SLB,SLE,SLV,SOM,SRB,STP,SUR,SVK,SVN,SWE,SWZ,SYR,TCD,TGO,THA,TJK,TKM,TLS,TON,TTO,TUN,TUR,TZA,UGA,UKR,URY,USA,UZB,VCT,VEN,VIR,VNM,VUT,WSM,YEM,ZAF,ZMB,ZWE'.split(',')
import random
random.shuffle(countries)
for iso3 in countries:
    print iso3
    try:
        F, causes = data.get_cod_data_all_causes(iso3=iso3)
        N, T, J = F.shape
        pi = pl.zeros((1000, T, J))
        for t in range(T):
            print t+1, 'of', T
            model, pi_t = models.fit_latent_simplex(F[:,t:(t+1),:])
            pi[:,t,:] = pi_t[:,0,:]
    except Exception, e:
        print e
        continue

    graphics.plot_F_and_pi(F, pi, causes, iso3)

    pl.savefig('/home/j/Project/Models/cod-correct/%s.png'%iso3)
