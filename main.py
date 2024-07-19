import sys
try:
 x = int(input('Choose task (Baseline:0, Gaussian:1, Lorentzian:2, Voigt:3, Gaussian Modified Scherrer:4, Lorentzian Modified Scherrer:5, Voigt Modified Scherrer:6, Gaussiaan Williamson–Hall:7, Lorentzian Williamson–Hall: 8,Voigt Williamson–Hall:9, ): '))
 if x == 0:
    print('You have chosen baseline.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import baseline
 elif x == 1:
    print('You have chosen XRD_GSM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_GSM
 elif x == 2:
    print('You have chosen XRD_LRM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_LRM
 elif x == 3:
    print('You have chosen XRD_VGLM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_VGLM
 elif x == 4:
    print('You have chosen XRD_GS_MSM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_GS_MSM
 elif x == 5:
    print('You have chosen XRD_LR_MSM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_LR_MSM
 elif x == 6:
    print('You have chosen XRD_VGL_MSM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_VGL_MSM
 elif x == 7:
    print('You have chosen XRD_GS_WHM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_GS_MSM
 elif x == 8:
    print('You have chosen XRD_LR_WHM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_LR_MSM
 elif x == 9:
    print('You have chosen XRD_VGL_WHM.py \nPlease go to the csv file and select the XRD spectrum to analyze!')
    import XRD_VGL_MSM
 else:
    print('You have not selected anything, please go to main function to run it again')
except:
 print("Finish!")