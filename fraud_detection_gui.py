

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import joblib  # For model loading
import pandas as pd
from tkinter import filedialog
import random
# ===== GUI APPLICATION =====

class CreditCardFraudDetectionGUI:
    """
    Main GUI application class for the Credit Card Fraud Detection System.
    Contains 28 PCA feature inputs plus Time and Amount.
    Uses a pretrained model.pkl for prediction.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Fraud Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Initialize model and feature list
        self.model = None
        self.features = []
        self.csv_tree = None 

        self.fraud_samples = [
            {
            'V1': '-0.769171691743673', 'V2': '1.3422122564916', 'V3': '-2.1714541366788', 'V4': '-0.151513299728476',
            'V5': '-0.648374467317641', 'V6': '-0.973504253142134', 'V7': '-1.70665765414268', 'V8': '0.313744516691369',
            'V9': '-1.98230184657376', 'V10': '-3.15812700566592', 'V11': '1.3415717221686', 'V12': '-3.29947236985172',
            'V13': '1.24764737362737', 'V14': '-6.39337284276668', 'V15': '-0.0532048016171774', 'V16': '-3.25804634128955',
            'V17': '-3.34889609796046', 'V18': '0.217330714330335', 'V19': '-0.917407954929339', 'V20': '-0.102294252090737',
            'V21': '-0.0361222946216654', 'V22': '-0.753591202726119', 'V23': '-0.0471133973013113', 'V24': '0.358492767536875',
            'V25': '-0.287406815695466', 'V26': '0.476505957022081', 'V27': '0.250530561357973', 'V28': '0.250987222920341',
            'Time': '125200.0', 'Amount': '40.0'
            },
            {
            'V1': '-4.59944700208203', 'V2': '2.76253994253031', 'V3': '-4.6565298706326', 'V4': '5.20140306725894',
            'V5': '-2.47038771213247', 'V6': '-0.357617930345621', 'V7': '-3.7671888274624', 'V8': '0.0614664440276634',
            'V9': '-1.83619986278079', 'V10': '-1.4706452575308', 'V11': '2.14393063330902', 'V12': '-5.83973605661705',
            'V13': '0.153011630876735', 'V14': '-6.17736456586248', 'V15': '-0.78541565373917', 'V16': '-4.51052273721062',
            'V17': '-10.3405128633486', 'V18': '-4.18117755349571', 'V19': '-0.376135682527478', 'V20': '-1.00065772232141',
            'V21': '1.58148035659457', 'V22': '0.261332719512975', 'V23': '0.621415022135499', 'V24': '0.994109926998982',
            'V25': '-0.687853106052664', 'V26': '-0.337531252843331', 'V27': '-1.61279104641522', 'V28': '1.23142473470435',
            'Time': '128519.0', 'Amount': '93.35'
            },
            {
            'V1': '-1.60021129907252', 'V2': '-3.48813018118561', 'V3': '-6.45930280667571', 'V4': '3.24681566349703',
            'V5': '-1.61460769860713', 'V6': '-1.26037454854632', 'V7': '0.288223321921388', 'V8': '-0.0489639910109505',
            'V9': '-0.734974617920653', 'V10': '-4.44148408070017', 'V11': '2.94437484242405', 'V12': '-3.80546898374331',
            'V13': '-2.10222683492697', 'V14': '-6.1061832734711', 'V15': '-0.641736399701777', 'V16': '-1.55596289626148',
            'V17': '-2.08406707082134', 'V18': '0.394247352869229', 'V19': '0.0833797227600835', 'V20': '3.18935489592753',
            'V21': '1.19117480721903', 'V22': '-0.96714144500067', 'V23': '-1.46342052446637', 'V24': '-0.624230851495284',
            'V25': '-0.176461532068143', 'V26': '0.400348351070197', 'V27': '0.152946864261687', 'V28': '0.477774924576796',
            'Time': '154278.0', 'Amount': '1504.93'
            },
            {
            'V1': '-17.4677100117887', 'V2': '10.1148157246654', 'V3': '-24.2021422329158', 'V4': '11.8054692105913',
            'V5': '-10.1980458075926', 'V6': '-2.5799380080012', 'V7': '-17.6567879964837', 'V8': '2.25690247596699',
            'V9': '-6.24214930949065', 'V10': '-12.8306571996417', 'V11': '9.44266526535108', 'V12': '-13.5474858999444',
            'V13': '0.960728558325579', 'V14': '-13.0287170264518', 'V15': '-0.426674498701321', 'V16': '-7.65266166506753',
            'V17': '-11.4853277896478', 'V18': '-4.72136957553102', 'V19': '0.550519004149565', 'V20': '1.00151850195952',
            'V21': '-2.32802441621057', 'V22': '0.940830319350178', 'V23': '1.29681706144136', 'V24': '-1.05510390359967',
            'V25': '0.11179202805362', 'V26': '0.679694612203418', 'V27': '2.09354057112569', 'V28': '-1.42549145377361',
            'Time': '21419.0', 'Amount': '1.0'
            },
            {
            'V1': '-1.23280412810524', 'V2': '2.24411874456383', 'V3': '-1.70382579275654', 'V4': '1.49253586483501',
            'V5': '-1.19298516971563', 'V6': '-1.68611021267902', 'V7': '-1.86461150490967', 'V8': '0.856122313785725',
            'V9': '-1.97353527775248', 'V10': '-3.94238290411726', 'V11': '2.40165068027699', 'V12': '-3.39426561777896',
            'V13': '-0.145161724970334', 'V14': '-4.32979551307589', 'V15': '0.758760885824274', 'V16': '-3.19904647594093',
            'V17': '-4.48018356526203', 'V18': '-2.48566878350317', 'V19': '-0.0958074336903356', 'V20': '0.207889301735382',
            'V21': '0.560475065574211', 'V22': '0.165681717805126', 'V23': '-0.0137535165177887', 'V24': '0.474935180269258',
            'V25': '-0.218725074414735', 'V26': '0.302809125708237', 'V27': '0.46603094153746', 'V28': '0.250133538809307',
            'Time': '47982.0', 'Amount': '0.76'
            },
            {
            'V1': '-4.72771265581559', 'V2': '3.04446910225824', 'V3': '-5.59835426695611', 'V4': '5.92819080241929',
            'V5': '-2.19076972938784', 'V6': '-1.52932296564747', 'V7': '-4.48742195988547', 'V8': '0.916391814103266',
            'V9': '-1.3070104231933', 'V10': '-4.13889121357616', 'V11': '5.14940878300581', 'V12': '-11.1240186070579',
            'V13': '0.543067765541171', 'V14': '-7.84094220494304', 'V15': '0.743633944785042', 'V16': '-6.77706923974191',
            'V17': '-9.93176515376766', 'V18': '-4.09302112201074', 'V19': '1.50492485905709', 'V20': '-0.207759445103882',
            'V21': '0.650988236282379', 'V22': '0.254983288954681', 'V23': '0.628843468802127', 'V24': '-0.23812845429317',
            'V25': '-0.671332331960485', 'V26': '-0.0335900627800998', 'V27': '-1.33177732223058', 'V28': '0.705697590023144',
            'Time': '12095.0', 'Amount': '30.39'
            },
            {
            'V1': '-16.5986647432584', 'V2': '10.5417508026636', 'V3': '-19.8189818085279', 'V4': '6.0172946473619',
            'V5': '-13.0259010501237', 'V6': '-4.12877900260013', 'V7': '-14.1188648469369', 'V8': '11.1611439952369',
            'V9': '-4.09955143059712', 'V10': '-9.22282550730978', 'V11': '6.32936467621872', 'V12': '-8.95219071159749',
            'V13': '-0.138363691274054', 'V14': '-9.82505442102721', 'V15': '0.0572235922374037', 'V16': '-7.54168742379503',
            'V17': '-14.2595985921417', 'V18': '-5.03505152375192', 'V19': '1.43226792970939', 'V20': '1.53491982113512',
            'V21': '1.72585282658181', 'V22': '-1.15160577552799', 'V23': '-0.680051901875725', 'V24': '0.108175914514646',
            'V25': '1.06687834149611', 'V26': '-0.233720431391328', 'V27': '1.70752055210726', 'V28': '0.51142316911239',
            'Time': '25231.0', 'Amount': '99.99'
            },
            {
            'V1': '-4.17167409700773', 'V2': '4.30844987710803', 'V3': '-8.23931473789338', 'V4': '4.16899712291352',
            'V5': '-3.85555178034389', 'V6': '-3.02305062433235', 'V7': '-5.52117091085174', 'V8': '3.04683514933838',
            'V9': '-2.6127598973363', 'V10': '-6.74604741416378', 'V11': '3.78934607143391', 'V12': '-7.68364279513021',
            'V13': '-1.18955114572977', 'V14': '-9.25752067304945', 'V15': '1.03638000621667', 'V16': '-5.13919711874594',
            'V17': '-8.43096726944362', 'V18': '-2.69604844926284', 'V19': '0.9013967736554', 'V20': '0.818595615438491',
            'V21': '0.902860571367263', 'V22': '-0.448209026783432', 'V23': '-0.0617210426827117', 'V24': '-0.0525628444096432',
            'V25': '0.0044867581738014', 'V26': '-0.293409430031766', 'V27': '1.28146640206865', 'V28': '0.307833804234966',
            'Time': '29234.0', 'Amount': '89.99'
            },
            {
            'V1': '-3.63280894927825', 'V2': '5.43726336227612', 'V3': '-9.13652148062595', 'V4': '10.3072263079132',
            'V5': '-5.42183029451154', 'V6': '-2.86481514933136', 'V7': '-10.6340876212042', 'V8': '3.01812657999175',
            'V9': '-4.89164032107494', 'V10': '-11.2350479111446', 'V11': '8.78878366710272', 'V12': '-18.5536970096458',
            'V13': '-0.339533407676443', 'V14': '-15.6231873302985', 'V15': '-0.188978574214972', 'V16': '-12.4279613630565',
            'V17': '-20.1590474539227', 'V18': '-6.88889109280947', 'V19': '2.58609321670737', 'V20': '1.35406479520345',
            'V21': '2.30988016889674', 'V22': '0.978660126299279', 'V23': '-0.0961301444473713', 'V24': '0.432376722544823',
            'V25': '-0.435627929953396', 'V26': '0.65089278643537', 'V27': '1.69360750804844', 'V28': '0.857685371687245',
            'Time': '93824.0', 'Amount': '8.54'
            },
            {
            'V1': '-0.676142670593205', 'V2': '1.1263660623459', 'V3': '-2.21369952308058', 'V4': '0.46830838758824',
            'V5': '-1.12054104443306', 'V6': '-0.0033462959955453', 'V7': '-2.23473929608742', 'V8': '1.21015796383769',
            'V9': '-0.65224992035649', 'V10': '-3.46389087904573', 'V11': '1.79496896856641', 'V12': '-2.77502154036273',
            'V13': '-0.418950143733104', 'V14': '-4.05716237716209', 'V15': '-0.712615968607599', 'V16': '-1.60301474745597',
            'V17': '-5.03532591722409', 'V18': '-0.50699988370237', 'V19': '0.266272320267649', 'V20': '0.247967752628092',
            'V21': '0.751825538154533', 'V22': '0.834107690036367', 'V23': '0.190943872773442', 'V24': '0.0320700856093905',
            'V25': '-0.739694822652264', 'V26': '0.47111096261267', 'V27': '0.385107448705084', 'V28': '0.19436147923645',
            'Time': '169351.0', 'Amount': '77.89'
            }

            # Add more if desired
        ]

        self.legit_samples = [
                    {
            'V1': '-0.451934279730244', 'V2': '0.0181323074526549', 'V3': '2.09051742325959', 'V4': '-0.26954406590558',
            'V5': '-0.21425872510581', 'V6': '0.89730284747048', 'V7': '0.108475586793103', 'V8': '0.195202771187845',
            'V9': '-1.50327856500953', 'V10': '0.645416341700527', 'V11': '1.34904478829361', 'V12': '0.225634247478362',
            'V13': '0.0350667351820448', 'V14': '-0.0389413673068581', 'V15': '0.675178946384842', 'V16': '-1.63862575141658',
            'V17': '-0.129245226709315', 'V18': '1.32936808943382', 'V19': '-1.64450061291887', 'V20': '-0.340895234645215',
            'V21': '-0.0487022958567397', 'V22': '0.442770764330415', 'V23': '0.0243878269360756', 'V24': '-0.333033837859025',
            'V25': '-0.491874009052153', 'V26': '-0.363779381974526', 'V27': '0.0345053288265341', 'V28': '-0.080639619649521',
            'Time': '41649.0', 'Amount': '57.0'
        },
        {
            'V1': '1.02933093384664', 'V2': '-0.472603049621391', 'V3': '1.18119781616255', 'V4': '0.806213953086176',
            'V5': '-1.19466420214704', 'V6': '0.007697644334343', 'V7': '-0.730361430158375', 'V8': '0.269428384090792',
            'V9': '0.838727663776334', 'V10': '-0.0635428918281535', 'V11': '0.770155854522916', 'V12': '0.521048193935192',
            'V13': '-1.17908552828832', 'V14': '0.031024435300999', 'V15': '-0.368218384166312', 'V16': '0.42048595385746',
            'V17': '-0.357908693363865', 'V18': '0.411808372862012', 'V19': '0.239258969539336', 'V20': '-0.0493380261134084',
            'V21': '-0.0193097818711413', 'V22': '-0.0806713582443383', 'V23': '-0.0172846467544072', 'V24': '0.336624782794712',
            'V25': '0.16477384682266', 'V26': '0.297532104022944', 'V27': '-0.0049453562110622', 'V28': '0.0243070425298522',
            'Time': '65478.0', 'Amount': '64.99'
        },
        {
            'V1': '-0.399349199608959', 'V2': '0.974621024487', 'V3': '1.84255935406719', 'V4': '-0.0424261836132531',
            'V5': '0.169126552601757', 'V6': '-0.54539972299312', 'V7': '0.935420641228258', 'V8': '-0.229532826734421',
            'V9': '-0.434562020454422', 'V10': '-0.182602738030989', 'V11': '0.381042549690946', 'V12': '0.601323371133524',
            'V13': '0.964294775337789', 'V14': '-0.223496533776105', 'V15': '0.99332534751791', 'V16': '-0.31898714727572',
            'V17': '-0.15951632161082', 'V18': '-1.05877172756644', 'V19': '-0.47286505462419', 'V20': '0.211810373842847',
            'V21': '-0.213345776769845', 'V22': '-0.309759828324062', 'V23': '0.0173172754877394', 'V24': '0.408416685650892',
            'V25': '-0.298273974736147', 'V26': '0.0484896901167432', 'V27': '0.122323869260814', 'V28': '-0.0992597736542823',
            'Time': '36350.0', 'Amount': '11.98'
        },
        {
            'V1': '-2.43447321443071', 'V2': '-1.49630092261442', 'V3': '0.988123243933406', 'V4': '-0.185585085496054',
            'V5': '1.63736629712212', 'V6': '-1.28897188033997', 'V7': '-1.05892166209401', 'V8': '0.69146823642827',
            'V9': '0.163342353480053', 'V10': '-1.16002343833251', 'V11': '-1.16989599544257', 'V12': '0.330159212543466',
            'V13': '0.25895474896615', 'V14': '0.386305625295146', 'V15': '0.734111911105668', 'V16': '0.438046735243131',
            'V17': '-0.52953227760122', 'V18': '0.15030146155238', 'V19': '-0.333336569232114', 'V20': '0.475120614509924',
            'V21': '0.166263692749722', 'V22': '-0.493615282569969', 'V23': '-0.0437927036812678', 'V24': '0.579518966227291',
            'V25': '-0.295912368361664', 'V26': '-0.660305404762921', 'V27': '0.0638148727303666', 'V28': '-0.278890680506288',
            'Time': '157783.0', 'Amount': '1.18'
        },
        {
            'V1': '1.65135543721592', 'V2': '-0.443478873848362', 'V3': '-1.5751797632959', 'V4': '0.707186538611032',
            'V5': '-0.304348149230406', 'V6': '-1.39893490269902', 'V7': '0.374425897574072', 'V8': '-0.357346954788866',
            'V9': '0.920151354423889', 'V10': '-0.935558341867779', 'V11': '0.000978440462552', 'V12': '0.674959729606591',
            'V13': '0.184438262861076', 'V14': '-1.45112519483065', 'V15': '-0.13480306208281', 'V16': '-0.197421484912326',
            'V17': '1.27192520486509', 'V18': '-0.224553936855981', 'V19': '-0.161311869006042', 'V20': '0.167449760941453',
            'V21': '-0.104914075860953', 'V22': '-0.401110572118278', 'V23': '0.0884925228741503', 'V24': '0.303968120699089',
            'V25': '-0.184790167945551', 'V26': '-0.130725623888015', 'V27': '-0.0328965070571373', 'V28': '0.0131373126928147',
            'Time': '161102.0', 'Amount': '181.54'
        },
        {
            'V1': '-1.24615005193395', 'V2': '0.863063907057315', 'V3': '0.478690331294345', 'V4': '0.405341886551499',
            'V5': '0.203485361233868', 'V6': '0.168487622534722', 'V7': '-0.136938542058818', 'V8': '0.682942053770281',
            'V9': '-2.00955859684143', 'V10': '-0.477579550809864', 'V11': '1.10649394547589', 'V12': '0.186051836688655',
            'V13': '0.610691006852801', 'V14': '-1.88621357091131', 'V15': '0.0646837713372074', 'V16': '-0.880417527161604',
            'V17': '1.71082073924038', 'V18': '2.98924327628777', 'V19': '-0.277784567129576', 'V20': '-0.0132114024205995',
            'V21': '-0.319464724548221', 'V22': '-0.588917811654062', 'V23': '-0.186106642367858', 'V24': '0.46803590265954',
            'V25': '0.571794405773225', 'V26': '-0.403076153712822', 'V27': '0.259077727501077', 'V28': '0.077266751307112',
            'Time': '145260.0', 'Amount': '56.0'
        },
        {
            'V1': '1.89611258616202', 'V2': '0.335755824193405', 'V3': '-0.761099671374196', 'V4': '3.82046851306475',
            'V5': '0.478209427241123', 'V6': '0.421634286602507', 'V7': '0.0068670471475165', 'V8': '0.0751272275782033',
            'V9': '-0.480066672079053', 'V10': '1.39064154099921', 'V11': '-1.75826654048758', 'V12': '-0.741531491703128',
            'V13': '-1.1393135801019', 'V14': '0.189678457586812', 'V15': '-1.31199770647224', 'V16': '0.503496633269285',
            'V17': '-0.460001096014891', 'V18': '-0.406087953678236', 'V19': '-1.3532227417165', 'V20': '-0.324933653397237',
            'V21': '0.0627967918176662', 'V22': '0.254140162782312', 'V23': '0.110492244587249', 'V24': '0.56945752192271',
            'V25': '0.113223443091966', 'V26': '0.0763194866227139', 'V27': '-0.0376853810503896', 'V28': '-0.0455275380096305',
            'Time': '123189.0', 'Amount': '19.0'
        },
        {
            'V1': '2.06636601900243', 'V2': '-1.26296332128172', 'V3': '-1.28855105178936', 'V4': '-1.21951452694383',
            'V5': '-0.470770105044877', 'V6': '0.285643979864078', 'V7': '-0.673463908289327', 'V8': '0.0347503100363156',
            'V9': '0.313988555771876', 'V10': '0.46045646657866', 'V11': '-0.305132934423667', 'V12': '1.11420565485506',
            'V13': '0.362875251577163', 'V14': '-0.37658622583272', 'V15': '-2.22682842162767', 'V16': '-2.05450155115375',
            'V17': '0.163436655908033', 'V18': '0.834763127113976', 'V19': '0.803651637226965', 'V20': '-0.443203567250389',
            'V21': '-0.717746449501638', 'V22': '-1.36338735034206', 'V23': '0.269576337157544', 'V24': '0.116423311770702',
            'V25': '-0.289046461049046', 'V26': '0.624574900745133', 'V27': '-0.0627858509471161', 'V28': '-0.062938649967648',
            'Time': '164061.0', 'Amount': '59.7'
        },
        {
            'V1': '-0.472462409201858', 'V2': '1.10640034968032', 'V3': '1.70068962703248', 'V4': '-0.0190777715779233',
            'V5': '0.0625765555443668', 'V6': '-0.582010260592369', 'V7': '0.7213597324417', 'V8': '-0.0199331765524273',
            'V9': '-0.610314827364836', 'V10': '-0.393185510395966', 'V11': '0.326369189288874', 'V12': '0.805082331338834',
            'V13': '1.1988895992295', 'V14': '-0.0580714031343229', 'V15': '0.939166985737417', 'V16': '-0.247639859343583',
            'V17': '-0.0673217545423756', 'V18': '-1.06107661649899', 'V19': '-0.475897897261351', 'V20': '0.131201979484562',
            'V21': '-0.165500160085131', 'V22': '-0.317732121559323', 'V23': '0.0301613679598905', 'V24': '0.415165240496032',
            'V25': '-0.268895687394548', 'V26': '0.0765304212212555', 'V27': '0.290818088681416', 'V28': '0.12338755962451',
            'Time': '42197.0', 'Amount': '2.18'
        },
        {
            'V1': '1.94393360037352', 'V2': '-0.733586311070298', 'V3': '-1.73554335857271', 'V4': '-0.783637448880311',
            'V5': '1.72874284008496', 'V6': '3.77791401770168', 'V7': '-1.13722278470802', 'V8': '1.02183396246113',
            'V9': '0.930679703774453', 'V10': '0.0007392951997238', 'V11': '-0.0227747963587755', 'V12': '0.427598037467011',
            'V13': '-0.0095455562602628', 'V14': '0.0918199082102817', 'V15': '0.95348215818276', 'V16': '0.371713080003444',
            'V17': '-0.598522204209487', 'V18': '-0.545065116860377', 'V19': '-0.205873272943131', 'V20': '-0.0659109766614475',
            'V21': '-0.205094318606091', 'V22': '-0.642173649209724', 'V23': '0.442298188327327', 'V24': '0.697921567129928',
            'V25': '-0.608343336659057', 'V26': '0.291924039920772', 'V27': '-0.0193745872928823', 'V28': '-0.0453549857908485',
            'Time': '122213.0', 'Amount': '30.0'
        }

            # Add more if desired
        ]
        
        # Attempt to load the pretrained model pipeline and feature list from dt_model.pkl
        try:
            loaded_obj = joblib.load('models/RF_model.pkl')  # Use your actual filename here
            self.model = loaded_obj.get('pipeline')
            self.features = loaded_obj.get('features', [])
            if self.model is None:
                messagebox.showerror("Model Load Error", "Model pipeline not found in dt_model.pkl")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.features = []

        # Use the loaded features list as feature_names for input fields if available,
        # otherwise default to V1-V28 + Time + Amount
        if self.features:
            self.feature_names = self.features
        else:
            self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

        self.entry_vars = {}  # Store StringVar objects for each input field

        self.create_widgets()
        
    def load_csv(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            # **RESET ALL PREVIOUS STATE FIRST**
            self.reset_csv_state()
            
            # Load new CSV
            df = pd.read_csv(filepath)
            
            # Check for missing columns
            missing = [c for c in self.features if c not in df.columns]
            if missing:
                messagebox.showerror("Missing Columns", f"CSV missing required columns: {missing}")
                return
            
            # Select only required features
            df = df[self.features]
            self.uploaded_df = df
            
            # Create fresh tree view
            self.create_csv_tree()
            
            # Populate tree with new data
            for _, row in df.iterrows():
                self.csv_tree.insert("", "end", values=list(row.values) + ["", ""])  # Empty prediction columns
            
            # Update UI state
            self.csv_predict_btn.config(state="normal")
            self.csv_download_btn.config(state="disabled")  # Reset download button
            self.summary_label.config(text=f"Loaded {len(df)} rows for prediction")
            
            # Reset filter to "All"
            self.csv_filter_var.set("All")
            
            messagebox.showinfo("Success", f"Successfully loaded {len(df)} transactions from CSV")
            
        except Exception as e:
            messagebox.showerror("CSV Load Error", f"Failed to load CSV: {str(e)}")
            self.reset_csv_state()  # Reset on error too
            
    def reset_csv_state(self):
        """Reset all CSV-related variables and UI elements to start fresh"""
        
        # Reset data variables
        self.uploaded_df = None
        if hasattr(self, 'predicted_df'):
            self.predicted_df = None
        
        # Destroy existing tree view
        if self.csv_tree:
            self.csv_tree.destroy()
            self.csv_tree = None
        
        # Reset UI elements
        self.csv_predict_btn.config(state="disabled")
        self.csv_download_btn.config(state="disabled")
        self.summary_label.config(text="")
        
        # Reset filter
        self.csv_filter_var.set("All")
        
        # Clear the tree frame
        for widget in self.csv_tree_frame.winfo_children():
            widget.destroy()

    def create_csv_tree(self):
        """Create a fresh CSV tree view"""
        
        # Create container for tree and scrollbars
        container = tk.Frame(self.csv_tree_frame)
        container.pack(fill="both", expand=True)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical")
        hsb = ttk.Scrollbar(container, orient="horizontal")
        
        # Create tree view
        self.csv_tree = ttk.Treeview(
            container, 
            columns=self.features + ["Prediction", "Fraud Probability"], 
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        
        # Configure scrollbars
        vsb.config(command=self.csv_tree.yview)
        hsb.config(command=self.csv_tree.xview)
        
        # Pack scrollbars
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.csv_tree.pack(side="left", fill="both", expand=True)
        
        # Configure column headings
        for col in self.features + ["Prediction", "Fraud Probability"]:
            self.csv_tree.heading(col, text=col)
            self.csv_tree.column(col, width=90, anchor="center")


    def create_widgets(self):
        # Title stays at the top
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        title_frame.pack_propagate(False)
        title_label = tk.Label(
            title_frame,
            text="🔒 Credit Card Fraud Detection System",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)

        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Tab 1: Transaction Input
        tab1 = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(tab1, text="Transaction Input")

        # Split the tab1 horizontally for inputs/results
        main_row = tk.Frame(tab1, bg='#f0f0f0')
        main_row.pack(fill='both', expand=True)

        left_panel = tk.LabelFrame(
            main_row,
            text="📝 Transaction Input",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        left_panel.pack(side='left', fill='y', expand=False, padx=(0, 15), pady=(10, 0))
        self.create_input_fields(left_panel)

        right_panel = tk.LabelFrame(
            main_row,
            text="📊 Analysis Results",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        right_panel.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=(10, 0))
        self.create_results_area(right_panel)

        # Control buttons in a row below panels
        buttons_row = tk.Frame(tab1, bg='#f0f0f0')
        buttons_row.pack(fill='x', padx=20, pady=15)
        self.create_buttons(buttons_row)

        # Tab 2: Predict from CSV 
        tab2 = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(tab2, text="Predict from CSV")
        desc = tk.Label(tab2,
                        text="CSV must contain columns: Time, V1 ... V28, Amount",
                        bg='#f0f0f0', font=('Arial', 12))
        desc.pack(pady=(10, 5))

        upload_btn = tk.Button(tab2, text="📂 Upload CSV", font=('Arial', 12),
                            bg='#3498db', fg='white', relief='raised',
                            command=self.load_csv)
        upload_btn.pack(pady=5)

        self.csv_tree_frame = tk.Frame(tab2)
        self.csv_tree_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.summary_label = tk.Label(tab2, text="", bg='#f0f0f0', font=('Arial', 12, 'bold'))
        self.summary_label.pack(pady=5)
        
        # Variable to track filter option
        self.csv_filter_var = tk.StringVar(value='All')

        filter_frame = tk.Frame(tab2, bg='#f0f0f0')
        filter_frame.pack(pady=5)

        tk.Label(filter_frame, text="🔍 Filter results: ", bg='#f0f0f0', font=('Arial', 11)).pack(side='left')

        options = ['All', 'Fraud only', 'Legit only']
        for opt in options:
            rb = ttk.Radiobutton(filter_frame, text=opt, value=opt, variable=self.csv_filter_var,
                                command=self.update_csv_filter)
            rb.pack(side='left', padx=5)


        self.csv_predict_btn = tk.Button(tab2, text="🔍 Predict Fraud from CSV", font=('Arial', 12),
                                        bg='#2ecc71', fg='white', relief='raised',
                                        state='disabled', command=self.predict_csv)
        self.csv_predict_btn.pack(pady=5)
        
                # Download button
        self.csv_download_btn = tk.Button(tab2, text="💾 Download CSV", font=('Arial', 12),
                                          bg='#f39c12', fg='white', relief='raised',
                                          state='disabled', command=self.download_csv)
        self.csv_download_btn.pack(pady=5)



    def create_input_fields(self, parent):
        """Create input fields for transaction features with descriptions"""
        descriptions = {}
        default_values = {}
        for i in range(1, 29):
            descriptions[f'V{i}'] = f'PCA Feature {i}'
            default_values[f'V{i}'] = '0.0'
        descriptions['Time'] = 'Seconds since first transaction'
        descriptions['Amount'] = 'Transaction amount ($)'
        default_values['Time'] = '3600'
        default_values['Amount'] = '150.50'

        #  scrollable area for inputs
        canvas = tk.Canvas(parent, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        #  input field for each feature
        for i, feature in enumerate(self.feature_names):
            # Feature label with description
            label_text = f"{feature}: {descriptions[feature]}"
            label = tk.Label(
                scrollable_frame,
                text=label_text,
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0',
                anchor='w'
            )
            label.grid(row=i * 2, column=0, sticky='w', padx=10, pady=(10, 2))

            # Entry field with default value
            var = tk.StringVar(value=default_values[feature])
            entry = tk.Entry(
                scrollable_frame,
                textvariable=var,
                font=('Arial', 11),
                width=25,
                relief='solid',
                borderwidth=1
            )
            entry.grid(row=i * 2 + 1, column=0, sticky='ew', padx=10, pady=(0, 5))

            self.entry_vars[feature] = var

        scrollable_frame.columnconfigure(0, weight=1)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_results_area(self, parent):
        """Create the results display area"""
        # Main prediction result
        self.result_label = tk.Label(
            parent,
            text="Ready for prediction...",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            wraplength=280
        )
        self.result_label.pack(pady=15)

        # Probability and risk level display
        self.prob_label = tk.Label(
            parent,
            text="",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#555555'
        )
        self.prob_label.pack(pady=5)

        # Visual progress bar for probability
        self.progress = ttk.Progressbar(
            parent,
            length=250,
            mode='determinate'
        )
        self.progress.pack(pady=10)

        # Detailed analysis area
        details_frame = tk.LabelFrame(
            parent,
            text="🔍 Analysis Details",
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        details_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.info_text = ScrolledText(
            details_frame,
            height=10,
            width=35,
            font=('Consolas', 9),
            bg='#ffffff',
            fg='#333333',
            relief='solid',
            borderwidth=1
        )
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)

    def create_buttons(self, parent):
        """Create control buttons"""
        # Main prediction button
        predict_btn = tk.Button(
            parent,
            text="🔍 Predict Fraud",
            command=self.predict_fraud,
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=25,
            pady=10
        )
        predict_btn.pack(side='left', padx=15)

        # Sample data buttons
        sample_legit_btn = tk.Button(
            parent,
            text="✅ Load Legitimate Sample",
            command=lambda: self.load_sample_data(False),
            font=('Arial', 11),
            bg='#2ecc71',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=15,
            pady=8
        )
        sample_legit_btn.pack(side='left', padx=5)

        sample_fraud_btn = tk.Button(
            parent,
            text="🚨 Load Fraud Sample",
            command=lambda: self.load_sample_data(True),
            font=('Arial', 11),
            bg='#e67e22',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=15,
            pady=8
        )
        sample_fraud_btn.pack(side='left', padx=5)

        # Utility buttons
        clear_btn = tk.Button(
            parent,
            text="🗑️ Clear All",
            command=self.clear_inputs,
            font=('Arial', 11),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=15,
            pady=8
        )
        clear_btn.pack(side='right', padx=15)

        help_btn = tk.Button(
            parent,
            text="❓ Help",
            command=self.show_help,
            font=('Arial', 11),
            bg='#9b59b6',
            fg='white',
            relief='raised',
            borderwidth=2,
            padx=15,
            pady=8
        )
        help_btn.pack(side='right', padx=5)

    def update_csv_filter(self):
        if not hasattr(self, 'predicted_df'):
            return

        filter_val = self.csv_filter_var.get()

        if filter_val == 'All':
            filtered = self.predicted_df
        elif filter_val == 'Fraud only':
            filtered = self.predicted_df[self.predicted_df['Prediction'] == 'Fraud']
        else:  # Legit only
            filtered = self.predicted_df[self.predicted_df['Prediction'] == 'Legit']

        self.update_csv_tree(filtered)

        self.summary_label.config(text=f"Showing {len(filtered)} rows out of {len(self.predicted_df)} total")

    
    def predict_fraud(self):
        """Main prediction function using model.pkl on user inputs"""
        if self.model is None:
            messagebox.showerror("Model Error", "No model loaded, cannot predict.")
            return

        try:
            # Collect input features in the order expected by the saved pipeline
            input_features = []
            for feat in self.features:
                val = float(self.entry_vars[feat].get())
                input_features.append(val)

            X = [input_features]
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]


            X = [input_features]  

            # Predict fraud or not and probability
            prediction = self.model.predict(X)[0]  # 0 or 1
            probability = self.model.predict_proba(X)[0][1]  # Probability of fraud class

            # Update progress bar and labels
            self.progress['value'] = probability * 100

            if prediction == 1:
                result_text = "🚨 FRAUD DETECTED!"
                result_color = '#e74c3c'
                risk_level = "HIGH RISK"
            else:
                result_text = "✅ TRANSACTION LEGITIMATE"
                result_color = '#2ecc71'
                risk_level = "LOW RISK"

            self.result_label.config(text=result_text, fg=result_color)
            self.prob_label.config(text=f"Fraud Probability: {probability:.1%}\nRisk Level: {risk_level}")

            # Display detailed analysis
            analysis = self.generate_analysis(input_features, prediction, probability)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, analysis)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    
    def predict_csv(self):
        if self.uploaded_df is None or self.model is None:
            messagebox.showerror("Prediction Error", "No CSV data loaded or model not loaded.")
            return

        try:
            X = self.uploaded_df.values

            preds = self.model.predict(X)
            proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(X)[:, 1]
                except Exception:
                    proba = None

            # Create new DataFrame with predictions and probabilities
            self.predicted_df = self.uploaded_df.copy()
            self.predicted_df['Prediction'] = ['Fraud' if p == 1 else 'Legit' for p in preds]
            if proba is not None:
                self.predicted_df['Fraud Probability'] = proba
            else:
                self.predicted_df['Fraud Probability'] = ''

            # Display all data initially (unfiltered)
            self.update_csv_tree(self.predicted_df)

            fraud_count = sum(preds == 1)
            legit_count = sum(preds == 0)
            total = len(preds)
            self.summary_label.config(text=f"Total: {total}  |  🛑 Fraud: {fraud_count}  |  ✅ Legitimate: {legit_count}")

            messagebox.showinfo("Prediction Complete", "Batch fraud predictions updated in the table.")
            self.csv_download_btn.config(state='normal')
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")

    def update_csv_tree(self, df):
        # Clear existing rows
        self.csv_tree.delete(*self.csv_tree.get_children())

        for _, row in df.iterrows():
            values = list(row[self.features])
            values.append(row.get('Prediction', ''))
            values.append(row.get('Fraud Probability', ''))
            self.csv_tree.insert('', 'end', values=values)
    
    def download_csv(self):
        """Download predicted CSV with filter applied"""
        if not hasattr(self, 'predicted_df'):
            messagebox.showerror("Download Error", "No predictions available to download.")
            return

        # Apply filter
        filter_val = self.csv_filter_var.get()
        if filter_val == 'All':
            df_to_save = self.predicted_df
        elif filter_val == 'Fraud only':
            df_to_save = self.predicted_df[self.predicted_df['Prediction'] == 'Fraud']
        else:
            df_to_save = self.predicted_df[self.predicted_df['Prediction'] == 'Legit']

        if df_to_save.empty:
            messagebox.showwarning("No Data", f"No rows match the filter: {filter_val}")
            return

        # Ask user where to save file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Predicted Results"
        )
        if not file_path:
            return

        try:
            df_to_save.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Predicted results saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file: {str(e)}")


    
    def generate_analysis(self, features, prediction, probability):
        """Generate detailed fraud analysis explanation based on model output"""
        analysis = "FRAUD DETECTION ANALYSIS\n"
        analysis += "=" * 35 + "\n\n"
        analysis += f"Prediction: {'FRAUD' if prediction else 'LEGITIMATE'}\n"
        analysis += f"Confidence: {probability:.1%}\n\n"

        # Show key PCA features with high absolute values
        extreme_features = []
        for i in range(28):
            val = features[i]
            if abs(val) > 2.5:
                extreme_features.append(f"V{i+1}({val:.2f})")

        if extreme_features:
            analysis += f"⚠️ Unusual PCA features: {', '.join(extreme_features)}\n"
        else:
            analysis += "✓ PCA features appear normal\n"

        # Time analysis
        time_val = features[-2]
        if time_val < 1000:
            analysis += "⚠️ Very early transaction time\n"
        elif time_val > 100000:
            analysis += "⚠️ Very late transaction time\n"
        else:
            analysis += "✓ Normal transaction timing\n"

        # Amount analysis
        amount_val = features[-1]
        if amount_val < 1:
            analysis += "⚠️ Extremely low transaction amount\n"
        elif amount_val > 5000:
            analysis += "⚠️ Very high transaction amount\n"
        elif amount_val < 10:
            analysis += "⚠️ Unusually small transaction amount\n"
        else:
            analysis += "✓ Transaction amount in normal range\n"

        # Risk summary
        analysis += "\nRISK ASSESSMENT:\n"
        if probability < 0.3:
            analysis += "🟢 Low Risk - Transaction appears normal\n"
        elif probability < 0.6:
            analysis += "🟡 Medium Risk - Some suspicious patterns\n"
        else:
            analysis += "🔴 High Risk - Multiple fraud indicators\n"

        return analysis

    def load_sample_data(self, is_fraud):
        """Load randomly selected predefined sample data"""
        if is_fraud:
            sample_data = random.choice(self.fraud_samples)
        else:
            sample_data = random.choice(self.legit_samples)

        for feature, value in sample_data.items():
            if feature in self.entry_vars:
                self.entry_vars[feature].set(value)

    def clear_inputs(self):
        """Reset all input fields and clear results"""
        for i in range(1, 29):
            self.entry_vars[f'V{i}'].set('0.0')
        self.entry_vars['Time'].set('3600')
        self.entry_vars['Amount'].set('150.50')

        self.result_label.config(text="Ready for prediction...", fg='#2c3e50')
        self.prob_label.config(text="")
        self.progress['value'] = 0
        self.info_text.delete(1.0, tk.END)

    def show_help(self):
        """Display comprehensive help information"""
        help_text = """CREDIT CARD FRAUD DETECTION SYSTEM HELP

🎯 PURPOSE:
This system analyzes credit card transactions using a pretrained machine learning model.

📝 HOW TO USE:
1. Enter transaction details (V1-V28, Time, Amount).
2. Click 'Predict Fraud' to analyze the transaction.
3. Review the results and detailed analysis.
4. Use sample data buttons for quick testing.

⚠️ INPUT FIELDS:
• V1-V28: PCA-transformed anonymized features (typically -3 to +3)
• Time: Seconds elapsed since first transaction of the day
• Amount: Transaction amount in USD

🎨 RESULTS:
• Green ✅: Legitimate (low fraud probability)
• Red 🚨: Fraud detected (high fraud probability)
• Progress bar indicates fraud likelihood (0-100%)
• Analysis details explain important indicators

📚 EDUCATIONAL PURPOSE:
This system demonstrates integration of machine learning models into a user-friendly GUI.
Real-world systems use much larger datasets and advanced models.

For questions or issues, check all fields have valid numbers."""

        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Fraud Detection System")
        help_window.geometry("600x500")
        help_window.configure(bg='#f0f0f0')
        help_window.resizable(False, False)

        help_window.transient(self.root)
        help_window.grab_set()

        text_widget = ScrolledText(
            help_window,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#333333',
            padx=15,
            pady=15,
            wrap=tk.WORD
        )
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert(1.0, help_text)
        text_widget.config(state='disabled')

        close_btn = tk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            font=('Arial', 11),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=5
        )
        close_btn.pack(pady=10)


# ===== APPLICATION LAUNCHER =====

def main():
    """Launch the Credit Card Fraud Detection GUI Application"""
    print("🚀 Launching Credit Card Fraud Detection System...")

    root = tk.Tk()
    root.resizable(True, True)
    root.minsize(800, 600)

    # Center window on screen
    window_width = 900
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    try:
        root.iconname("Fraud Detection")
    except:
        pass

    app = CreditCardFraudDetectionGUI(root)

    print("✅ Application launched successfully!")
    print("📋 Ready for fraud detection analysis...")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n🛑 Application terminated by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        print("👋 Application closed")


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("=" * 60)
    print("🔒 CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("\n📋 SYSTEM COMPONENTS:")
    print("✅ ML-based fraud detection with pretrained model.pkl")
    print("✅ Interactive Tkinter GUI interface")
    print("✅ Input validation and error handling")
    print("✅ Sample data for testing")
    print("✅ Detailed analysis and explanations")
    print("✅ Comprehensive help system")

    print("\n🎓 EDUCATIONAL FEATURES:")
    print("• Demonstrates ML model integration into GUI")
    print("• Shows feature importance and analysis")
    print("• Provides hands-on GUI development experience")
    print("• Illustrates real-world application development")

   

    print("\n🚀 Starting application...")
    main()
