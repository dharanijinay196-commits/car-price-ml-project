# car-price-ml-project
Machine Learning project for predicting car price category using Python
car-price-ml-project
│
├── data
│   └── car_data.csv
│
├── notebooks
│   ├── data_cleaning.ipynb
│   └── ml_model.ipynb
│
└── README.md
,vehicle_uid,name,year,selling_price,km_driven,fuel,seller_type,transmission,owner,car_record_id,car_age,price_category,km_category,owner_count,fuel_transmission,brand,price_per_km,is_diesel
0,CAR_000001,Maruti 800 AC,2007,60000.0,70000.0,Petrol,Individual,Manual,First Owner,1,18,Low,Medium,0.0,Petrol_Manual,Maruti,0.86,0
1,CAR_000002,Maruti Wagon R LXI Minor,2007,135000.0,50000.0,Petrol,Individual,Manual,First Owner,2,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.7,0
2,CAR_000003,Hyundai Verna 1.6 SX,2012,600000.0,100000.0,Diesel,Individual,Manual,First Owner,3,13,Low,Medium,0.0,Diesel_Manual,Hyundai,6.0,1
3,CAR_000004,Datsun RediGO T Option,2017,250000.0,46000.0,Petrol,Individual,Manual,First Owner,4,8,Low,Medium,0.0,Petrol_Manual,Datsun,5.43,0
4,CAR_000005,Honda Amaze VX i-DTEC,2014,450000.0,141000.0,Diesel,Individual,Manual,Second Owner,5,11,Low,High,0.0,Diesel_Manual,Honda,3.19,1
5,CAR_000006,Maruti Alto LX BSIII,2007,140000.0,125000.0,Petrol,Individual,Manual,First Owner,6,18,Low,High,0.0,Petrol_Manual,Maruti,1.12,0
6,CAR_000007,Hyundai Xcent 1.2 Kappa S,2016,550000.0,25000.0,Petrol,Individual,Manual,First Owner,7,9,Mid,Low,0.0,Petrol_Manual,Hyundai,22.0,0
7,CAR_000008,Tata Indigo Grand Petrol,2014,240000.0,60000.0,Petrol,Individual,Manual,Second Owner,8,11,Low,Medium,0.0,Petrol_Manual,Tata,4.0,0
8,CAR_000009,Hyundai Creta 1.6 VTVT S,2015,4461000.0,25000.0,Diesel,Individual,Manual,First Owner,9,10,High,Low,0.0,Petrol_Manual,Hyundai,34.0,0
9,CAR_000010,Maruti Celerio Green VXI,2017,365000.0,78000.0,CNG,Individual,Manual,First Owner,10,8,Mid,High,0.0,CNG_Manual,Maruti,4.68,0
10,CAR_000011,Chevrolet Sail 1.2 Base,2015,260000.0,35000.0,Petrol,Individual,Manual,First Owner,11,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,7.43,0
11,CAR_000012,Tata Indigo Grand Petrol,2014,250000.0,100000.0,Petrol,Individual,Manual,First Owner,12,11,Low,High,0.0,Petrol_Manual,Tata,2.5,0
12,CAR_000013,Toyota Corolla Altis 1.8 VL CVT,2018,1650000.0,25000.0,Petrol,Dealer,Automatic,First Owner,13,7,Premium,Low,0.0,Petrol_Automatic,Toyota,66.0,0
13,CAR_000014,Maruti 800 AC,2007,60000.0,70000.0,Petrol,Individual,Manual,First Owner,14,18,Low,Medium,0.0,Petrol_Manual,Maruti,0.86,0
14,CAR_000015,Maruti Wagon R LXI Minor,2007,135000.0,60000.0,Petrol,Individual,Manual,First Owner,15,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.7,0
15,CAR_000016,Hyundai Verna 1.6 SX,2012,600000.0,100000.0,CNG,Individual,Manual,First Owner,16,13,Mid,High,0.0,Diesel_Manual,Hyundai,6.0,1
16,CAR_000017,Datsun RediGO T Option,2017,250000.0,46000.0,Petrol,Individual,Manual,First Owner,17,8,Low,Medium,0.0,Petrol_Manual,Datsun,5.43,0
17,CAR_000018,Honda Amaze VX i-DTEC,2014,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,18,11,Mid,High,0.0,Diesel_Manual,Honda,3.19,1
18,CAR_000019,Maruti Alto LX BSIII,2007,140000.0,125000.0,Petrol,Individual,Manual,First Owner,19,18,Low,High,0.0,Petrol_Manual,Maruti,1.12,0
19,CAR_000020,Hyundai Xcent 1.2 Kappa S,2016,550000.0,25000.0,Petrol,Individual,Manual,First Owner,20,9,Mid,Low,0.0,Petrol_Manual,Hyundai,22.0,0
20,CAR_000021,Tata Indigo Grand Petrol,2014,240000.0,60000.0,Petrol,Individual,Manual,Second Owner,21,11,Low,Medium,0.0,Petrol_Manual,Tata,4.0,0
21,CAR_000022,Hyundai Creta 1.6 VTVT S,2015,850000.0,25000.0,Petrol,Individual,Manual,First Owner,22,10,High,Low,0.0,Petrol_Manual,Hyundai,34.0,0
22,CAR_000023,Maruti Celerio Green VXI,2017,365000.0,60000.0,CNG,Individual,Manual,First Owner,23,8,Mid,High,0.0,CNG_Manual,Maruti,4.68,0
23,CAR_000024,Chevrolet Sail 1.2 Base,2015,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,24,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,7.43,0
24,CAR_000025,Tata Indigo Grand Petrol,2014,250000.0,100000.0,Petrol,Individual,Manual,First Owner,25,11,Low,High,0.0,Petrol_Manual,Tata,2.5,0
25,CAR_000026,Toyota Corolla Altis 1.8 VL CVT,2018,1650000.0,25000.0,Petrol,Dealer,Automatic,First Owner,26,7,Premium,Low,0.0,Petrol_Automatic,Toyota,66.0,0
26,CAR_000027,Maruti Ciaz VXi Plus,2015,585000.0,24000.0,Petrol,Dealer,Manual,First Owner,27,10,Mid,Low,0.0,Petrol_Manual,Maruti,24.38,0
27,CAR_000028,Hyundai Venue SX Opt Diesel,2019,1195000.0,5000.0,Diesel,Dealer,Manual,First Owner,28,6,Premium,Low,0.0,Diesel_Manual,Hyundai,239.0,1
28,CAR_000029,Chevrolet Enjoy TCDi LTZ 7 Seater,2013,390000.0,33000.0,Diesel,Individual,Manual,Second Owner,29,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,11.82,1
29,CAR_000030,Jaguar XF 2.2 Litre Luxury,2014,1964999.0,28000.0,Diesel,Dealer,Automatic,First Owner,30,11,Premium,Low,0.0,Diesel_Automatic,Jaguar,70.18,1
30,CAR_000031,Mercedes-Benz New C-Class 220 CDI AT,2013,1425000.0,59000.0,Diesel,Dealer,Automatic,First Owner,31,12,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,24.15,1
31,CAR_000032,Maruti Vitara Brezza ZDi Plus AMT,2018,975000.0,4500.0,Diesel,Dealer,Automatic,First Owner,32,7,High,Low,0.0,Diesel_Automatic,Maruti,216.67,1
32,CAR_000033,Audi Q5 2.0 TDI,2011,1190000.0,60000.0,Diesel,Dealer,Manual,First Owner,33,14,Premium,Very High,0.0,Diesel_Automatic,Audi,6.77,1
33,CAR_000034,Honda City V MT,2018,4461000.0,60000.0,Petrol,Dealer,Manual,First Owner,34,7,High,Low,0.0,Petrol_Manual,Honda,64.14,0
34,CAR_000035,Tata Tigor 1.2 Revotron XT,2018,525000.0,15000.0,Petrol,Individual,Manual,First Owner,35,7,Mid,Low,0.0,Petrol_Manual,Tata,35.0,0
35,CAR_000036,Audi A6 2.0 TDI  Design Edition,2013,1735000.0,50000.0,Diesel,Dealer,Automatic,First Owner,36,12,Premium,Medium,0.0,Diesel_Automatic,Audi,34.7,1
36,CAR_000037,Mercedes-Benz New C-Class C 220 CDI Avantgarde,2012,1375000.0,33800.0,Diesel,Dealer,Automatic,Second Owner,37,13,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,40.68,1
37,CAR_000038,Skoda Superb Ambition 2.0 TDI CR AT,2011,450000.0,60000.0,Diesel,Dealer,Automatic,Second Owner,38,14,Mid,High,0.0,Diesel_Automatic,Skoda,3.45,1
38,CAR_000039,Toyota Corolla Altis G AT,2016,900000.0,50000.0,Petrol,Individual,Automatic,First Owner,39,9,High,Medium,0.0,Petrol_Automatic,Toyota,18.0,0
39,CAR_000040,Toyota Innova 2.5 G (Diesel) 7 Seater,2015,1300000.0,80000.0,Diesel,Individual,Manual,First Owner,40,10,Premium,High,0.0,Diesel_Manual,Toyota,16.25,1
40,CAR_000041,Jeep Compass 1.4 Sport Plus BSIV,2019,1400000.0,10000.0,Petrol,Individual,Manual,First Owner,41,6,Premium,Medium,0.0,Petrol_Manual,Jeep,140.0,0
41,CAR_000042,Mercedes-Benz E-Class E 200 CGI Elegance,2010,850000.0,119000.0,Petrol,Dealer,Automatic,First Owner,42,15,High,High,0.0,Petrol_Automatic,Mercedes-Benz,7.14,0
42,CAR_000043,Hyundai i10 Magna 1.1L,2014,229999.0,60000.0,LPG,Individual,Manual,Fourth & Above Owner,43,11,Low,Medium,0.0,Petrol_Manual,Hyundai,3.83,0
43,CAR_000044,BMW 3 Series 320d Sport Line,2013,1550000.0,75800.0,Diesel,Dealer,Automatic,Second Owner,44,12,Premium,High,0.0,Diesel_Automatic,BMW,20.45,1
44,CAR_000045,Audi Q7 35 TDI Quattro Premium,2009,1250000.0,78000.0,Diesel,Dealer,Manual,Third Owner,45,16,Premium,High,0.0,Diesel_Automatic,Audi,16.03,1
45,CAR_000046,Hyundai Elantra CRDi S,2012,625000.0,40000.0,Diesel,Individual,Manual,First Owner,46,13,High,Medium,0.0,Diesel_Manual,Hyundai,15.62,1
46,CAR_000047,Mahindra Scorpio 1.99 S10,2014,1050000.0,50000.0,Diesel,Individual,Manual,First Owner,47,11,Premium,Medium,0.0,Diesel_Manual,Mahindra,21.0,1
47,CAR_000048,Honda City i DTEC V,2014,560000.0,74000.0,Electric,Individual,Manual,Second Owner,48,11,Low,High,0.0,Diesel_Manual,Honda,7.57,1
48,CAR_000049,Maruti Wagon R VXI BS IV with ABS,2014,290000.0,64000.0,Petrol,Individual,Manual,Second Owner,49,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.53,0
49,CAR_000050,Maruti Wagon R VXI BS IV,2012,275000.0,60000.0,Petrol,Individual,Manual,Second Owner,50,13,Low,Medium,0.0,Petrol_Manual,Maruti,4.58,0
50,CAR_000051,Mahindra Scorpio LX,2009,411000.0,120000.0,Diesel,Individual,Manual,Second Owner,51,16,Mid,High,0.0,Diesel_Manual,Mahindra,3.42,1
51,CAR_000052,Hyundai Santro Xing GLS,2009,4461000.0,79000.0,Petrol,Individual,Manual,Third Owner,52,16,Low,High,0.0,Petrol_Manual,Hyundai,1.9,0
52,CAR_000053,Hyundai Grand i10 1.2 Kappa Asta,2019,500000.0,15000.0,CNG,Individual,Manual,First Owner,53,6,Mid,Medium,0.0,Petrol_Manual,Hyundai,33.33,0
53,CAR_000054,Maruti Alto LXi,2006,100000.0,80000.0,Petrol,Individual,Manual,First Owner,54,19,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
54,CAR_000055,Maruti Swift Dzire VDI Optional,2017,725000.0,18500.0,Diesel,Dealer,Manual,First Owner,55,8,High,Low,0.0,Diesel_Manual,Maruti,39.19,1
55,CAR_000056,Maruti Eeco 5 Seater AC BSIV,2018,401000.0,10200.0,Petrol,Dealer,Manual,First Owner,56,7,Mid,Low,0.0,Petrol_Manual,Maruti,39.31,0
56,CAR_000057,Hyundai i20 Sportz 1.4 CRDi,2018,750000.0,29000.0,Diesel,Dealer,Manual,First Owner,57,7,Low,Low,0.0,Diesel_Manual,Hyundai,25.86,1
57,CAR_000058,Maruti Omni Maruti Omni MPI STD BSIII 5-STR W/ IMMOBILISER,2018,310000.0,28000.0,Petrol,Dealer,Manual,First Owner,58,7,Mid,Low,0.0,Petrol_Manual,Maruti,11.07,0
58,CAR_000059,Maruti Swift ZDi BSIV,2016,665000.0,46000.0,Diesel,Dealer,Manual,First Owner,59,9,High,Medium,0.0,Diesel_Manual,Maruti,14.46,1
59,CAR_000060,Hyundai i20 2015-2017 Sportz Option 1.4 CRDi,2014,465000.0,70000.0,Diesel,Dealer,Manual,First Owner,60,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,6.64,1
60,CAR_000061,Maruti Alto LXi,2012,160000.0,60000.0,Petrol,Individual,Manual,Second Owner,61,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.67,0
61,CAR_000062,Mahindra Jeep CL 500 MDI,1996,250000.0,35000.0,Diesel,Individual,Manual,Second Owner,62,29,Low,Medium,0.0,Diesel_Manual,Mahindra,7.14,1
62,CAR_000063,Honda City i DTEC VX,2014,675000.0,90000.0,Diesel,Dealer,Manual,First Owner,63,11,High,High,0.0,Diesel_Manual,Honda,7.5,1
63,CAR_000064,Maruti Wagon R VXI BS IV,2013,300000.0,80000.0,Petrol,Dealer,Manual,First Owner,64,12,Low,High,0.0,Petrol_Manual,Maruti,3.75,0
64,CAR_000065,Tata Indica DLS,2005,70000.0,80000.0,Diesel,Individual,Manual,First Owner,65,20,Low,High,0.0,Diesel_Manual,Tata,0.88,1
65,CAR_000066,Hyundai EON Magna Plus,2014,240000.0,73300.0,Petrol,Dealer,Manual,Second Owner,66,11,Low,High,0.0,Petrol_Manual,Hyundai,3.27,0
66,CAR_000067,Toyota Etios GD,2014,525000.0,92000.0,Diesel,Dealer,Manual,First Owner,67,11,Mid,High,0.0,Diesel_Manual,Toyota,5.71,1
67,CAR_000068,Maruti Alto LXi,2009,151000.0,66764.0,Petrol,Dealer,Manual,Second Owner,68,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.26,0
68,CAR_000069,Maruti Alto LXi,2009,140000.0,100000.0,Petrol,Individual,Manual,First Owner,69,16,Low,High,0.0,Petrol_Manual,Maruti,1.4,0
69,CAR_000070,Chevrolet Tavera Neo LS B3 - 7(C) seats BSIII,2010,280000.0,350000.0,Diesel,Individual,Manual,Second Owner,70,15,Low,Very High,0.0,Diesel_Manual,Chevrolet,0.8,1
70,CAR_000071,Toyota Corolla Altis Diesel D4DG,2011,4461000.0,230000.0,Diesel,Individual,Manual,First Owner,71,14,Mid,Very High,0.0,Diesel_Manual,Toyota,1.52,1
71,CAR_000072,Mahindra Scorpio 1.99 S6 Plus,2017,570000.0,60000.0,Diesel,Individual,Manual,First Owner,72,8,Mid,Medium,0.0,Diesel_Manual,Mahindra,9.5,1
72,CAR_000073,Hyundai EON Magna Plus,2018,300000.0,31000.0,LPG,Individual,Manual,First Owner,73,7,Low,Medium,0.0,Petrol_Manual,Hyundai,9.68,0
73,CAR_000074,Tata Indigo Classic Dicor,2007,100000.0,39000.0,Diesel,Individual,Manual,First Owner,74,18,Low,Medium,0.0,Diesel_Manual,Tata,2.56,1
74,CAR_000075,Toyota Innova 2.5 V Diesel 8-seater,2009,500000.0,120000.0,Diesel,Individual,Manual,Third Owner,75,16,Mid,Medium,0.0,Diesel_Manual,Toyota,4.17,1
75,CAR_000076,Tata Indica Vista Quadrajet LS,2014,125000.0,166000.0,Diesel,Individual,Manual,Fourth & Above Owner,76,11,Low,Very High,0.0,Diesel_Manual,Tata,0.75,1
76,CAR_000077,Maruti Swift 1.3 VXi,2006,130000.0,110000.0,Petrol,Individual,Manual,Third Owner,77,19,Low,High,0.0,Petrol_Manual,Maruti,1.18,0
77,CAR_000078,Ford EcoSport 1.5 Diesel Titanium BSIV,2017,925000.0,35000.0,Diesel,Individual,Manual,First Owner,78,8,High,Medium,0.0,Diesel_Manual,Ford,26.43,1
78,CAR_000079,Maruti Ciaz 1.3 Delta,2018,750000.0,60000.0,Electric,Individual,Manual,First Owner,79,7,High,Medium,0.0,Diesel_Manual,Maruti,12.5,1
79,CAR_000080,Honda Civic 1.8 V AT,2007,200000.0,54000.0,Petrol,Individual,Automatic,Second Owner,80,18,Low,Medium,0.0,Petrol_Automatic,Honda,3.7,0
80,CAR_000081,Hyundai i10 Sportz 1.2,2010,4461000.0,60000.0,Petrol,Individual,Manual,Second Owner,81,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.94,0
81,CAR_000082,Skoda Rapid 1.5 TDI Elegance,2014,450000.0,120000.0,Diesel,Individual,Manual,Second Owner,82,11,Mid,High,0.0,Diesel_Manual,Skoda,3.75,1
82,CAR_000083,Hyundai Getz GLS,2005,80000.0,120000.0,Petrol,Individual,Manual,Third Owner,83,20,Low,Medium,0.0,Petrol_Manual,Hyundai,0.67,0
83,CAR_000084,Nissan Terrano XL,2014,650000.0,76000.0,Petrol,Individual,Manual,Second Owner,84,11,High,High,0.0,Petrol_Manual,Nissan,8.55,0
84,CAR_000085,Hyundai Grand i10 CRDi Sportz,2015,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,85,10,Mid,High,0.0,Diesel_Manual,Hyundai,5.62,1
85,CAR_000086,Hyundai Elite i20 Diesel Era,2019,650000.0,25000.0,Diesel,Individual,Manual,First Owner,86,6,Low,Low,0.0,Diesel_Manual,Hyundai,26.0,1
86,CAR_000087,Honda Amaze S i-VTEC,2016,495000.0,11958.0,Petrol,Dealer,Manual,First Owner,87,9,Mid,Low,0.0,Petrol_Manual,Honda,41.39,0
87,CAR_000088,Honda Brio S MT,2015,371000.0,20000.0,Petrol,Dealer,Manual,First Owner,88,10,Mid,Low,0.0,Petrol_Manual,Honda,18.55,0
88,CAR_000089,Hyundai Creta 1.6 SX Option,2017,1025000.0,60000.0,Petrol,Dealer,Manual,First Owner,89,8,Premium,Low,0.0,Petrol_Manual,Hyundai,113.89,0
89,CAR_000090,Mercedes-Benz S-Class S 350d Connoisseurs Edition,2017,8150000.0,6500.0,Diesel,Dealer,Automatic,First Owner,90,8,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,1253.85,1
90,CAR_000091,Mahindra XUV500 W8 2WD,2015,750000.0,70000.0,Diesel,Individual,Manual,First Owner,91,10,High,Medium,0.0,Diesel_Manual,Mahindra,10.71,1
91,CAR_000092,Renault Duster 85PS Diesel RxL Optional,2013,600000.0,120000.0,Diesel,Individual,Manual,First Owner,92,12,Mid,High,0.0,Diesel_Manual,Renault,5.0,1
92,CAR_000093,Hyundai Santro Xing XO,2007,80000.0,58000.0,Petrol,Dealer,Manual,Second Owner,93,18,Low,Medium,0.0,Petrol_Manual,Hyundai,1.38,0
93,CAR_000094,Mahindra Bolero 2011-2019 SLE,2013,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,94,12,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.23,1
94,CAR_000095,Audi A6 2.0 TDI Premium Plus,2014,1470000.0,34000.0,Diesel,Dealer,Automatic,Second Owner,95,11,Premium,Medium,0.0,Diesel_Automatic,Audi,43.24,1
95,CAR_000096,Fiat Avventura MULTIJET Emotion,2015,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,96,10,Mid,Medium,0.0,Diesel_Manual,Fiat,6.6,1
96,CAR_000097,Audi A8 4.2 TDI,2013,4461000.0,49000.0,Diesel,Dealer,Automatic,First Owner,97,12,Premium,Medium,0.0,Diesel_Automatic,Audi,57.14,1
97,CAR_000098,Datsun RediGO 1.0 S,2017,210000.0,60000.0,Petrol,Dealer,Manual,Second Owner,98,8,Low,Low,0.0,Petrol_Manual,Datsun,14.0,0
98,CAR_000099,Volkswagen Jetta 1.4 TSI Comfortline,2013,450000.0,50000.0,Petrol,Individual,Manual,First Owner,99,12,Mid,Medium,0.0,Petrol_Manual,Volkswagen,9.0,0
99,CAR_000100,Audi A4 2.0 TDI 177 Bhp Premium Plus,2013,1150000.0,53000.0,Diesel,Dealer,Automatic,First Owner,100,12,Premium,Medium,0.0,Diesel_Automatic,Audi,21.7,1
100,CAR_000101,Honda Civic 1.8 V AT,2009,210000.0,63500.0,Petrol,Dealer,Automatic,First Owner,101,16,Low,Medium,0.0,Petrol_Automatic,Honda,3.31,0
101,CAR_000102,Mercedes-Benz E-Class Exclusive E 200 BSIV,2018,4500000.0,9800.0,Petrol,Dealer,Manual,First Owner,102,7,Premium,Low,0.0,Petrol_Automatic,Mercedes-Benz,459.18,0
102,CAR_000103,BMW X1 sDrive 20d xLine,2017,2750000.0,13000.0,Diesel,Individual,Automatic,First Owner,103,8,Premium,Low,0.0,Diesel_Automatic,BMW,211.54,1
103,CAR_000104,Volvo V40 D3 R Design,2018,1975000.0,21000.0,Diesel,Dealer,Automatic,First Owner,104,7,Premium,Low,0.0,Diesel_Automatic,Volvo,94.05,1
104,CAR_000105,Maruti SX4 Zxi BSIII,2008,175000.0,29173.0,Petrol,Dealer,Manual,First Owner,105,17,Low,Low,0.0,Petrol_Manual,Maruti,6.0,0
105,CAR_000106,BMW 7 Series 730Ld,2012,2500000.0,48000.0,Diesel,Dealer,Automatic,First Owner,106,13,Premium,Medium,0.0,Diesel_Automatic,BMW,52.08,1
106,CAR_000107,Mahindra Bolero Power Plus SLX,2017,628000.0,120000.0,Diesel,Individual,Manual,First Owner,107,8,High,High,0.0,Diesel_Manual,Mahindra,5.23,1
107,CAR_000108,Hyundai Sonata CRDi M/T,2010,600000.0,100000.0,Diesel,Individual,Manual,First Owner,108,15,Mid,High,0.0,Diesel_Manual,Hyundai,6.0,1
108,CAR_000109,Nissan Micra Active XV Petrol,2017,399000.0,30000.0,Petrol,Individual,Manual,First Owner,109,8,Mid,Low,0.0,Petrol_Manual,Nissan,13.3,0
109,CAR_000110,Mahindra Xylo D4,2018,4461000.0,87000.0,Diesel,Individual,Manual,First Owner,110,7,High,High,0.0,Diesel_Manual,Mahindra,8.62,1
110,CAR_000111,Hyundai Elite i20 Sportz Plus Dual Tone BSIV,2019,750000.0,15000.0,Petrol,Individual,Manual,First Owner,111,6,High,Low,0.0,Petrol_Manual,Hyundai,50.0,0
111,CAR_000112,Renault KWID RXT,2017,315000.0,16000.0,Petrol,Individual,Manual,First Owner,112,8,Mid,Low,0.0,Petrol_Manual,Renault,19.69,0
112,CAR_000113,Mahindra Xylo E4 BS III,2012,600000.0,60000.0,Diesel,Individual,Manual,First Owner,113,13,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.0,1
113,CAR_000114,Maruti Wagon R LXI Minor,2010,100000.0,60000.0,CNG,Individual,Manual,Fourth & Above Owner,114,15,Low,Medium,0.0,Petrol_Manual,Maruti,1.67,0
114,CAR_000115,Maruti SX4 ZXI MT BSIV,2011,250000.0,110000.0,Petrol,Individual,Manual,Third Owner,115,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.27,0
115,CAR_000116,Renault KWID RXT,2017,350000.0,25000.0,Petrol,Individual,Manual,First Owner,116,8,Mid,Low,0.0,Petrol_Manual,Renault,14.0,0
116,CAR_000117,Hyundai Creta 1.4 CRDi S,2016,780000.0,60000.0,Diesel,Individual,Manual,First Owner,117,9,High,Medium,0.0,Diesel_Manual,Hyundai,13.0,1
117,CAR_000118,Maruti Swift Dzire VDI,2014,434000.0,79350.0,Diesel,Individual,Manual,Second Owner,118,11,Mid,High,0.0,Diesel_Manual,Maruti,5.47,1
118,CAR_000119,Hyundai Verna 1.6 VTVT AT S Option,2016,690000.0,80000.0,Petrol,Individual,Manual,First Owner,119,9,High,High,0.0,Petrol_Automatic,Hyundai,8.62,0
119,CAR_000120,Mahindra Scorpio LX BSIV,2014,555000.0,90000.0,Diesel,Individual,Manual,Second Owner,120,11,Low,Medium,0.0,Diesel_Manual,Mahindra,6.17,1
120,CAR_000121,Maruti SX4 Vxi BSIII,2007,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,121,18,Low,High,0.0,Petrol_Manual,Maruti,1.33,0
121,CAR_000122,Maruti Ertiga VDI,2014,500000.0,120000.0,Diesel,Individual,Manual,Second Owner,122,11,Mid,High,0.0,Diesel_Manual,Maruti,4.17,1
122,CAR_000123,Chevrolet Beat Diesel,2013,165000.0,60000.0,Diesel,Individual,Manual,Second Owner,123,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.75,1
123,CAR_000124,Maruti Zen LX,2004,95000.0,60000.0,Petrol,Individual,Manual,First Owner,124,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.9,0
124,CAR_000125,Maruti Baleno Delta 1.2,2018,550000.0,20000.0,Petrol,Individual,Manual,First Owner,125,7,Mid,Low,0.0,Petrol_Manual,Maruti,27.5,0
125,CAR_000126,Maruti Swift Vdi BSIII,2007,100000.0,110000.0,Diesel,Individual,Manual,Fourth & Above Owner,126,18,Low,High,0.0,Diesel_Manual,Maruti,0.91,1
126,CAR_000127,Tata Nano Lx BSIV,2012,100000.0,50000.0,Petrol,Individual,Manual,Second Owner,127,13,Low,Medium,0.0,Petrol_Manual,Tata,2.0,0
127,CAR_000128,Toyota Innova 2.5 GX (Diesel) 8 Seater,2012,500000.0,100000.0,Diesel,Individual,Manual,First Owner,128,13,Mid,High,0.0,Diesel_Manual,Toyota,5.0,1
128,CAR_000129,Maruti Ertiga SHVS VDI,2016,800000.0,70000.0,LPG,Individual,Manual,Second Owner,129,9,High,Medium,0.0,Diesel_Manual,Maruti,11.43,1
129,CAR_000130,Hyundai Creta 1.6 CRDi SX,2016,840000.0,70000.0,Diesel,Individual,Manual,Second Owner,130,9,Low,Medium,0.0,Diesel_Manual,Hyundai,12.0,1
130,CAR_000131,Honda Amaze S i-Vtech,2016,490000.0,60000.0,Petrol,Individual,Manual,First Owner,131,9,Low,Medium,0.0,Petrol_Manual,Honda,9.8,0
131,CAR_000132,Tata Indica Vista Aqua 1.4 TDI,2010,125000.0,81000.0,Diesel,Individual,Manual,Second Owner,132,15,Low,High,0.0,Diesel_Manual,Tata,1.54,1
132,CAR_000133,Chevrolet Tavera Neo 2 LS B4 7 Str BSIII,2012,4461000.0,120000.0,Diesel,Individual,Manual,Third Owner,133,13,Mid,High,0.0,Diesel_Manual,Chevrolet,3.33,1
133,CAR_000134,Chevrolet Cruze LTZ,2015,1000000.0,3600.0,Diesel,Dealer,Manual,First Owner,134,10,High,Low,0.0,Diesel_Manual,Chevrolet,277.78,1
134,CAR_000135,Ford Figo Aspire 1.2 Ti-VCT Titanium Plus,2015,4461000.0,14272.0,Petrol,Dealer,Manual,First Owner,135,10,Mid,Low,0.0,Petrol_Manual,Ford,37.14,0
135,CAR_000136,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2017,840000.0,49213.0,Electric,Dealer,Manual,First Owner,136,8,High,Medium,0.0,Diesel_Manual,Ford,17.07,1
136,CAR_000137,Hyundai i10 Sportz 1.1L,2015,229999.0,40000.0,Petrol,Individual,Manual,First Owner,137,10,Low,Medium,0.0,Petrol_Manual,Hyundai,5.75,0
137,CAR_000138,Maruti 800 Std,1998,40000.0,40000.0,Petrol,Individual,Manual,Fourth & Above Owner,138,27,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
138,CAR_000139,Chevrolet Spark 1.0 LS,2012,130000.0,80000.0,Petrol,Individual,Manual,First Owner,139,13,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.62,0
139,CAR_000140,Hyundai EON Era Plus,2015,200000.0,70000.0,Petrol,Individual,Manual,First Owner,140,10,Low,Medium,0.0,Petrol_Manual,Hyundai,2.86,0
140,CAR_000141,Tata Indica Vista Aqua TDI BSIII,2011,120000.0,70000.0,Diesel,Individual,Manual,First Owner,141,14,Low,Medium,0.0,Diesel_Manual,Tata,1.71,1
141,CAR_000142,Hyundai Santro LP zipPlus,2003,75000.0,60000.0,Petrol,Individual,Manual,First Owner,142,22,Low,Medium,0.0,Petrol_Manual,Hyundai,1.32,0
142,CAR_000143,Tata Bolt Quadrajet XE,2016,250000.0,120000.0,Diesel,Individual,Manual,First Owner,143,9,Low,High,0.0,Diesel_Manual,Tata,2.08,1
143,CAR_000144,Maruti 800 AC BSIII,2005,100000.0,30000.0,Petrol,Individual,Manual,First Owner,144,20,Low,Low,0.0,Petrol_Manual,Maruti,3.33,0
144,CAR_000145,Hyundai EON Era Plus,2013,4461000.0,3240.0,Petrol,Individual,Manual,Second Owner,145,12,Low,Low,0.0,Petrol_Manual,Hyundai,86.42,0
145,CAR_000146,Hyundai i20 Magna 1.2,2015,540000.0,5000.0,Petrol,Individual,Manual,First Owner,146,10,Mid,Low,0.0,Petrol_Manual,Hyundai,108.0,0
146,CAR_000147,Hyundai i20 1.2 Asta,2018,700000.0,10000.0,Petrol,Individual,Manual,First Owner,147,7,High,Low,0.0,Petrol_Manual,Hyundai,70.0,0
147,CAR_000148,Maruti Ciaz VDi Plus,2015,525000.0,100000.0,Diesel,Individual,Manual,First Owner,148,10,Mid,High,0.0,Diesel_Manual,Maruti,5.25,1
148,CAR_000149,Hyundai i20 Asta 1.4 CRDi,2016,430000.0,80000.0,Diesel,Individual,Manual,First Owner,149,9,Mid,High,0.0,Diesel_Manual,Hyundai,5.38,1
149,CAR_000150,Hyundai Santro LE,2002,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,150,23,Low,Medium,0.0,Petrol_Manual,Hyundai,0.93,0
150,CAR_000151,Maruti Vitara Brezza VDi,2018,4461000.0,35000.0,Diesel,Individual,Manual,First Owner,151,7,High,Medium,0.0,Diesel_Manual,Maruti,22.86,1
151,CAR_000152,Hyundai Santro Xing XL eRLX Euro III,2007,4461000.0,114000.0,Petrol,Individual,Manual,Second Owner,152,18,Low,High,0.0,Petrol_Manual,Hyundai,0.66,0
152,CAR_000153,Hyundai Getz 1.3 GLS,2008,4461000.0,53772.0,Diesel,Individual,Manual,First Owner,153,17,Low,Medium,0.0,Petrol_Manual,Hyundai,3.91,0
153,CAR_000154,Mahindra Quanto C8,2012,195000.0,140000.0,Diesel,Individual,Manual,Second Owner,154,13,Low,High,0.0,Diesel_Manual,Mahindra,1.39,1
154,CAR_000155,Chevrolet Tavera Neo 3 LS 7 C BSIII,2015,400000.0,120000.0,Diesel,Individual,Manual,First Owner,155,10,Mid,High,0.0,Diesel_Manual,Chevrolet,3.33,1
155,CAR_000156,Hyundai EON Magna Plus,2012,170000.0,60000.0,Petrol,Individual,Manual,First Owner,156,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.83,0
156,CAR_000157,Renault KWID RXT,2016,225000.0,25000.0,Petrol,Individual,Manual,First Owner,157,9,Low,Low,0.0,Petrol_Manual,Renault,9.0,0
157,CAR_000158,Maruti Wagon R DUO LPG,2014,4461000.0,90000.0,LPG,Individual,Manual,First Owner,158,11,Low,High,0.0,LPG_Manual,Maruti,2.33,0
158,CAR_000159,Maruti Wagon R LXI,2020,240000.0,120000.0,Petrol,Individual,Manual,First Owner,159,5,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
159,CAR_000160,Chevrolet Enjoy 1.3 TCDi LS 8,2015,300000.0,60000.0,Diesel,Individual,Manual,First Owner,160,10,Low,Very High,0.0,Diesel_Manual,Chevrolet,1.71,1
160,CAR_000161,Chevrolet Spark 1.0 LS,2011,99000.0,100000.0,CNG,Individual,Manual,Third Owner,161,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,0.99,0
161,CAR_000162,Honda City i VTEC SV,2014,620000.0,36000.0,Petrol,Individual,Manual,First Owner,162,11,High,Medium,0.0,Petrol_Manual,Honda,17.22,0
162,CAR_000163,Honda Amaze VX i-DTEC,2013,500000.0,30000.0,Diesel,Individual,Manual,First Owner,163,12,Low,Low,0.0,Diesel_Manual,Honda,16.67,1
163,CAR_000164,Jaguar XJ 5.0 L V8 Supercharged,2010,2550000.0,40000.0,Petrol,Individual,Manual,Second Owner,164,15,Premium,Medium,0.0,Petrol_Automatic,Jaguar,63.75,0
164,CAR_000165,Honda Brio E MT,2013,260000.0,70000.0,Petrol,Individual,Manual,Second Owner,165,12,Low,Medium,0.0,Petrol_Manual,Honda,3.71,0
165,CAR_000166,Maruti Swift VVT ZXI,2017,550000.0,60000.0,Petrol,Individual,Manual,First Owner,166,8,Mid,Medium,0.0,Petrol_Manual,Maruti,9.17,0
166,CAR_000167,Tata Indigo CR4,2011,4461000.0,155500.0,Diesel,Individual,Manual,First Owner,167,14,Low,Very High,0.0,Diesel_Manual,Tata,0.96,1
167,CAR_000168,Hyundai i10 Asta AT,2011,350000.0,50000.0,Petrol,Individual,Automatic,Second Owner,168,14,Mid,Medium,0.0,Petrol_Automatic,Hyundai,7.0,0
168,CAR_000169,Chevrolet Beat LT,2016,320000.0,40000.0,Petrol,Individual,Manual,First Owner,169,9,Mid,Medium,0.0,Petrol_Manual,Chevrolet,8.0,0
169,CAR_000170,Maruti Swift VDI BSIV,2015,400000.0,100000.0,Diesel,Individual,Manual,Third Owner,170,10,Mid,High,0.0,Diesel_Manual,Maruti,4.0,1
170,CAR_000171,Maruti Alto LXi,2012,175000.0,80000.0,Petrol,Individual,Manual,Third Owner,171,13,Low,High,0.0,Petrol_Manual,Maruti,2.19,0
171,CAR_000172,Chevrolet Beat LT,2011,120000.0,90000.0,Petrol,Individual,Manual,Second Owner,172,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.33,0
172,CAR_000173,Renault Duster 110PS Diesel RxZ,2013,320000.0,90000.0,Diesel,Individual,Manual,Second Owner,173,12,Mid,High,0.0,Diesel_Manual,Renault,3.56,1
173,CAR_000174,Hyundai Santro Xing XG,2006,70000.0,110000.0,Petrol,Individual,Manual,Third Owner,174,19,Low,High,0.0,Petrol_Manual,Hyundai,0.64,0
174,CAR_000175,Maruti Swift Dzire ZXI Plus,2019,810000.0,15000.0,Petrol,Individual,Manual,First Owner,175,6,High,Low,0.0,Petrol_Manual,Maruti,54.0,0
175,CAR_000176,Maruti 800 AC,2007,4461000.0,100000.0,Petrol,Individual,Manual,Second Owner,176,18,Low,High,0.0,Petrol_Manual,Maruti,0.95,0
176,CAR_000177,Maruti Alto K10 LXI CNG,2020,282000.0,40000.0,CNG,Individual,Manual,First Owner,177,5,Low,Medium,0.0,CNG_Manual,Maruti,7.05,0
177,CAR_000178,Maruti 800 Std BSII,2004,80000.0,60000.0,Petrol,Individual,Manual,Second Owner,178,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.33,0
178,CAR_000179,Tata Nano LX SE,2013,72000.0,25000.0,Petrol,Individual,Manual,First Owner,179,12,Low,Low,0.0,Petrol_Manual,Tata,2.88,0
179,CAR_000180,Hyundai i20 Magna 1.2,2016,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,180,9,Mid,Low,0.0,Petrol_Manual,Hyundai,24.0,0
180,CAR_000181,Skoda Rapid 1.6 MPI Ambition With Alloy Wheel,2015,640000.0,23000.0,Petrol,Individual,Manual,First Owner,181,10,Low,Low,0.0,Petrol_Manual,Skoda,27.83,0
181,CAR_000182,Maruti Alto K10 VXI,2018,380000.0,22155.0,Petrol,Individual,Manual,First Owner,182,7,Mid,Low,0.0,Petrol_Manual,Maruti,17.15,0
182,CAR_000183,Maruti Ciaz 1.4 Delta,2018,650000.0,60000.0,LPG,Individual,Manual,First Owner,183,7,High,Medium,0.0,Petrol_Manual,Maruti,9.29,0
183,CAR_000184,Maruti Alto LX,2009,150000.0,60000.0,Petrol,Individual,Manual,Second Owner,184,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
184,CAR_000185,Hyundai i20 Asta,2009,4461000.0,110000.0,Petrol,Individual,Manual,Second Owner,185,16,Low,High,0.0,Petrol_Manual,Hyundai,2.55,0
185,CAR_000186,Tata Nexon 1.2 Revotron XM,2018,430000.0,60000.0,Petrol,Individual,Manual,First Owner,186,7,Mid,Low,0.0,Petrol_Manual,Tata,14.33,0
186,CAR_000187,Mahindra Bolero Power Plus SLX,2019,800000.0,15000.0,Diesel,Individual,Manual,First Owner,187,6,High,Medium,0.0,Diesel_Manual,Mahindra,53.33,1
187,CAR_000188,Maruti Zen D,2003,75000.0,100000.0,Diesel,Individual,Manual,Second Owner,188,22,Low,High,0.0,Diesel_Manual,Maruti,0.75,1
188,CAR_000189,Volkswagen Vento Celeste 1.5 TDI Highline AT,2016,4461000.0,70000.0,Diesel,Individual,Automatic,First Owner,189,9,High,Medium,0.0,Diesel_Automatic,Volkswagen,9.29,1
189,CAR_000190,Maruti Eeco 7 Seater Standard BSIV,2017,390000.0,60000.0,Petrol,Individual,Manual,First Owner,190,8,Mid,Low,0.0,Petrol_Manual,Maruti,16.25,0
190,CAR_000191,Honda City 1.5 EXI,2005,100000.0,120000.0,Petrol,Individual,Manual,Second Owner,191,20,Low,High,0.0,Petrol_Manual,Honda,0.83,0
191,CAR_000192,Mercedes-Benz New C-Class 220 CDI AT,2013,1500000.0,35000.0,Diesel,Individual,Automatic,First Owner,192,12,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,42.86,1
192,CAR_000193,Maruti SX4 Zxi with Leather BSIII,2007,4461000.0,78380.0,Petrol,Individual,Manual,First Owner,193,18,Low,High,0.0,Petrol_Manual,Maruti,2.23,0
193,CAR_000194,Renault Duster 110PS Diesel RxZ,2012,434999.0,110000.0,Diesel,Individual,Manual,Second Owner,194,13,Mid,High,0.0,Diesel_Manual,Renault,3.95,1
194,CAR_000195,Ford Figo Diesel Titanium,2010,170000.0,90000.0,Diesel,Individual,Manual,Third Owner,195,15,Low,High,0.0,Diesel_Manual,Ford,1.89,1
195,CAR_000196,Ford Figo Diesel Titanium,2011,190000.0,90000.0,Electric,Individual,Manual,Third Owner,196,14,Low,High,0.0,Diesel_Manual,Ford,2.11,1
196,CAR_000197,Maruti Swift Dzire VDi,2009,4461000.0,150000.0,Diesel,Individual,Manual,Third Owner,197,16,Low,High,0.0,Diesel_Manual,Maruti,1.67,1
197,CAR_000198,Mahindra Xylo E4,2009,229999.0,60000.0,Diesel,Individual,Manual,Third Owner,198,16,Low,Very High,0.0,Diesel_Manual,Mahindra,1.0,1
198,CAR_000199,Maruti Esteem Vxi - BSIII,2006,140000.0,70000.0,Petrol,Individual,Manual,Third Owner,199,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
199,CAR_000200,Maruti Swift VDI BSIV,2017,600000.0,80362.0,Diesel,Individual,Manual,First Owner,200,8,Mid,High,0.0,Diesel_Manual,Maruti,7.47,1
200,CAR_000201,Hyundai i20 1.2 Sportz,2012,280000.0,110000.0,Petrol,Individual,Manual,Third Owner,201,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.55,0
201,CAR_000202,Chevrolet Beat Diesel LT,2012,150000.0,60000.0,Diesel,Individual,Manual,First Owner,202,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.5,1
202,CAR_000203,Chevrolet Cruze LTZ AT,2015,650000.0,50000.0,Diesel,Individual,Automatic,First Owner,203,10,High,Medium,0.0,Diesel_Automatic,Chevrolet,13.0,1
203,CAR_000204,Nissan Micra XL,2014,210000.0,40000.0,Petrol,Individual,Manual,Second Owner,204,11,Low,Medium,0.0,Petrol_Manual,Nissan,5.25,0
204,CAR_000205,BMW 5 Series 520d Luxury Line,2017,2900000.0,40000.0,Diesel,Individual,Automatic,First Owner,205,8,Low,Medium,0.0,Diesel_Automatic,BMW,72.5,1
205,CAR_000206,Chevrolet Enjoy 1.3 TCDi LS 8,2013,4461000.0,20000.0,Diesel,Dealer,Manual,First Owner,206,12,Mid,Low,0.0,Diesel_Manual,Chevrolet,21.25,1
206,CAR_000207,Hyundai EON Era Plus Option,2014,265000.0,55000.0,Petrol,Dealer,Manual,First Owner,207,11,Low,Medium,0.0,Petrol_Manual,Hyundai,4.82,0
207,CAR_000208,Fiat Linea T Jet Emotion,2018,890000.0,1136.0,Petrol,Dealer,Manual,First Owner,208,7,High,Low,0.0,Petrol_Manual,Fiat,783.45,0
208,CAR_000209,Renault Scala RxL,2013,365000.0,55000.0,CNG,Dealer,Manual,First Owner,209,12,Mid,Medium,0.0,Petrol_Manual,Renault,6.64,0
209,CAR_000210,Ford Figo Petrol Titanium,2014,350000.0,43000.0,Petrol,Dealer,Manual,First Owner,210,11,Mid,Medium,0.0,Petrol_Manual,Ford,8.14,0
210,CAR_000211,Maruti Ciaz ZDi SHVS,2016,685000.0,60000.0,Diesel,Individual,Manual,First Owner,211,9,Low,Medium,0.0,Diesel_Manual,Maruti,11.42,1
211,CAR_000212,Skoda Rapid 1.5 TDI Ambition BSIV,2018,4461000.0,2650.0,Diesel,Dealer,Manual,First Owner,212,7,High,Low,0.0,Diesel_Manual,Skoda,354.72,1
212,CAR_000213,Mahindra XUV500 W6 2WD,2017,1000000.0,60000.0,Diesel,Individual,Manual,First Owner,213,8,High,Medium,0.0,Diesel_Manual,Mahindra,25.0,1
213,CAR_000214,Mahindra XUV300 W8 Option,2019,1150000.0,15000.0,Petrol,Dealer,Manual,First Owner,214,6,Premium,Low,0.0,Petrol_Manual,Mahindra,76.67,0
214,CAR_000215,Maruti Ertiga SHVS ZDI,2016,450000.0,115962.0,Diesel,Individual,Manual,Second Owner,215,9,Low,High,0.0,Diesel_Manual,Maruti,3.88,1
215,CAR_000216,Nissan Terrano XE D,2015,590000.0,65000.0,Diesel,Dealer,Manual,First Owner,216,10,Mid,Medium,0.0,Diesel_Manual,Nissan,9.08,1
216,CAR_000217,Maruti S-Cross Facelift,2017,800000.0,40000.0,Diesel,Individual,Manual,First Owner,217,8,High,Medium,0.0,Diesel_Manual,Maruti,20.0,1
217,CAR_000218,Hyundai i20 Magna 1.4 CRDi (Diesel),2012,385000.0,58000.0,Diesel,Dealer,Manual,First Owner,218,13,Low,Medium,0.0,Diesel_Manual,Hyundai,6.64,1
218,CAR_000219,Mercedes-Benz New C-Class C 220 CDI BE Avantgare,2013,2000000.0,15000.0,LPG,Individual,Automatic,First Owner,219,12,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,133.33,1
219,CAR_000220,Hyundai EON Era Plus,2012,235000.0,54000.0,Petrol,Dealer,Manual,First Owner,220,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.35,0
220,CAR_000221,Volkswagen Ameo 1.5 TDI Highline,2017,4461000.0,56000.0,Diesel,Dealer,Manual,First Owner,221,8,High,Medium,0.0,Diesel_Manual,Volkswagen,11.43,1
221,CAR_000222,Maruti Alto LX,2006,52000.0,60000.0,Petrol,Individual,Manual,Third Owner,222,19,Low,Medium,0.0,Petrol_Manual,Maruti,0.43,0
222,CAR_000223,Maruti Omni LPG CARGO BSIII W IMMOBILISER,2009,80000.0,60000.0,LPG,Individual,Manual,Second Owner,223,16,Low,High,0.0,LPG_Manual,Maruti,0.89,0
223,CAR_000224,Maruti Alto LX,2012,140000.0,70000.0,Petrol,Individual,Manual,Second Owner,224,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
224,CAR_000225,Hyundai Verna i (Petrol),2008,120000.0,90000.0,Petrol,Individual,Manual,Second Owner,225,17,Low,High,0.0,Petrol_Manual,Hyundai,1.33,0
225,CAR_000226,Mahindra Renault Logan 1.5 DLS,2008,89999.0,213000.0,Diesel,Individual,Manual,First Owner,226,17,Low,Very High,0.0,Diesel_Manual,Mahindra,0.42,1
226,CAR_000227,Chevrolet Optra Magnum 2.0 LS BSIII,2012,180000.0,120000.0,Diesel,Individual,Manual,First Owner,227,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.5,1
227,CAR_000228,Mahindra Scorpio S11 BSIV,2017,1500000.0,20000.0,Diesel,Individual,Manual,First Owner,228,8,Premium,Low,0.0,Diesel_Manual,Mahindra,75.0,1
228,CAR_000229,Hyundai i20 Active 1.2 SX,2017,700000.0,10000.0,Petrol,Individual,Manual,First Owner,229,8,High,Low,0.0,Petrol_Manual,Hyundai,70.0,0
229,CAR_000230,Maruti SX4 Celebration Diesel,2012,285000.0,80000.0,Diesel,Individual,Manual,Second Owner,230,13,Low,Medium,0.0,Diesel_Manual,Maruti,3.56,1
230,CAR_000231,Hyundai Grand i10 Magna,2015,390000.0,90000.0,Petrol,Individual,Manual,First Owner,231,10,Mid,High,0.0,Petrol_Manual,Hyundai,4.33,0
231,CAR_000232,Maruti Alto LXi BSIII,2008,125000.0,80000.0,Petrol,Individual,Manual,First Owner,232,17,Low,High,0.0,Petrol_Manual,Maruti,1.56,0
232,CAR_000233,Maruti SX4 Zxi with Leather BSIII,2008,225000.0,60000.0,Petrol,Individual,Manual,Second Owner,233,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.75,0
233,CAR_000234,Hyundai i10 Era,2011,175000.0,60000.0,Petrol,Individual,Manual,Second Owner,234,14,Low,High,0.0,Petrol_Manual,Hyundai,1.26,0
234,CAR_000235,Toyota Innova 2.5 V Diesel 7-seater,2011,1075000.0,160000.0,Diesel,Individual,Manual,Second Owner,235,14,Premium,Very High,0.0,Diesel_Manual,Toyota,6.72,1
235,CAR_000236,Honda Mobilio V i DTEC,2014,300000.0,150000.0,Diesel,Individual,Manual,First Owner,236,11,Low,High,0.0,Diesel_Manual,Honda,2.0,1
236,CAR_000237,Toyota Innova 2.5 G (Diesel) 7 Seater BS IV,2010,700000.0,163000.0,Diesel,Individual,Manual,Fourth & Above Owner,237,15,High,Very High,0.0,Diesel_Manual,Toyota,4.29,1
237,CAR_000238,Tata Indica V2 2001-2011 DLS BSIII,2009,90000.0,120000.0,Diesel,Dealer,Manual,First Owner,238,16,Low,High,0.0,Diesel_Manual,Tata,0.75,1
238,CAR_000239,Tata Indica Vista TDI LS,2011,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,239,14,Low,High,0.0,Diesel_Manual,Tata,1.0,1
239,CAR_000240,Chevrolet Beat Diesel LS,2012,220000.0,60000.0,Diesel,Individual,Manual,Second Owner,240,13,Low,High,0.0,Diesel_Manual,Chevrolet,2.2,1
240,CAR_000241,Tata Zest Quadrajet 1.3 75PS XE,2015,300000.0,140000.0,Electric,Dealer,Manual,First Owner,241,10,Low,High,0.0,Diesel_Manual,Tata,2.14,1
241,CAR_000242,Skoda Fabia 1.2L Diesel Ambiente,2010,180000.0,87000.0,Diesel,Dealer,Manual,Second Owner,242,15,Low,High,0.0,Diesel_Manual,Skoda,2.07,1
242,CAR_000243,Maruti 800 Std,2013,170000.0,70000.0,Petrol,Individual,Manual,First Owner,243,12,Low,Medium,0.0,Petrol_Manual,Maruti,2.43,0
243,CAR_000244,Ford Figo Petrol Titanium,2015,300000.0,50000.0,Petrol,Individual,Manual,Second Owner,244,10,Low,Medium,0.0,Petrol_Manual,Ford,6.0,0
244,CAR_000245,Toyota Innova 2.5 VX (Diesel) 8 Seater,2014,1050000.0,70000.0,Diesel,Individual,Manual,First Owner,245,11,Premium,Medium,0.0,Diesel_Manual,Toyota,15.0,1
245,CAR_000246,Honda Mobilio V i DTEC,2014,300000.0,60000.0,Diesel,Individual,Manual,First Owner,246,11,Low,High,0.0,Diesel_Manual,Honda,2.0,1
246,CAR_000247,Hyundai Grand i10 1.2 CRDi Asta,2018,465000.0,25000.0,Diesel,Individual,Manual,First Owner,247,7,Mid,Low,0.0,Diesel_Manual,Hyundai,18.6,1
247,CAR_000248,Chevrolet Beat Diesel LT,2011,130000.0,80000.0,Petrol,Individual,Manual,Second Owner,248,14,Low,High,0.0,Diesel_Manual,Chevrolet,1.62,1
248,CAR_000249,Datsun GO Plus T Option Petrol,2018,434999.0,10000.0,Petrol,Individual,Manual,First Owner,249,7,Mid,Low,0.0,Petrol_Manual,Datsun,43.5,0
249,CAR_000250,Hyundai Grand i10 1.2 Kappa Asta,2018,500000.0,32000.0,Petrol,Individual,Manual,First Owner,250,7,Mid,Medium,0.0,Petrol_Manual,Hyundai,15.62,0
250,CAR_000251,Maruti Omni MPI STD BSIV,2018,200000.0,10000.0,Diesel,Individual,Manual,First Owner,251,7,Low,Low,0.0,Petrol_Manual,Maruti,20.0,0
251,CAR_000252,Maruti Baleno Alpha 1.2,2017,625000.0,52000.0,CNG,Dealer,Manual,First Owner,252,8,High,Medium,0.0,Petrol_Manual,Maruti,12.02,0
252,CAR_000253,Ford Fiesta Classic 1.4 SXI Duratorq,2006,110000.0,120000.0,Diesel,Individual,Manual,Third Owner,253,19,Low,High,0.0,Diesel_Manual,Ford,0.92,1
253,CAR_000254,Hyundai Elite i20 Asta Option BSIV,2019,800000.0,11240.0,Petrol,Individual,Manual,First Owner,254,6,High,Low,0.0,Petrol_Manual,Hyundai,71.17,0
254,CAR_000255,Hyundai Grand i10 CRDi Magna,2017,4461000.0,66000.0,Diesel,Dealer,Manual,First Owner,255,8,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.42,1
255,CAR_000256,Maruti Ertiga SHVS ZDI,2017,880000.0,64000.0,Diesel,Dealer,Manual,First Owner,256,8,High,Medium,0.0,Diesel_Manual,Maruti,13.75,1
256,CAR_000257,Hyundai Santro Xing GL Plus,2013,290000.0,49000.0,Petrol,Individual,Manual,First Owner,257,12,Low,Medium,0.0,Petrol_Manual,Hyundai,5.92,0
257,CAR_000258,Tata Sumo GX TC 7 Str BSIII,2006,115999.0,100000.0,Diesel,Individual,Manual,Second Owner,258,19,Low,High,0.0,Diesel_Manual,Tata,1.16,1
258,CAR_000259,Renault KWID RXT,2018,360000.0,26500.0,Petrol,Individual,Manual,First Owner,259,7,Low,Low,0.0,Petrol_Manual,Renault,13.58,0
259,CAR_000260,Maruti 800 AC,2002,65000.0,100000.0,Petrol,Individual,Manual,Second Owner,260,23,Low,High,0.0,Petrol_Manual,Maruti,0.65,0
260,CAR_000261,Maruti Vitara Brezza LDi Option,2017,685000.0,72000.0,Diesel,Dealer,Manual,First Owner,261,8,High,High,0.0,Diesel_Manual,Maruti,9.51,1
261,CAR_000262,Honda Jazz S,2009,300000.0,50000.0,Petrol,Individual,Manual,First Owner,262,16,Low,Medium,0.0,Petrol_Manual,Honda,6.0,0
262,CAR_000263,Hyundai i20 1.4 Sportz,2017,680000.0,44000.0,LPG,Dealer,Manual,First Owner,263,8,High,Medium,0.0,Diesel_Manual,Hyundai,15.45,1
263,CAR_000264,Maruti Ertiga SHVS ZDI Plus,2017,1000000.0,60000.0,Diesel,Individual,Manual,Second Owner,264,8,High,Medium,0.0,Diesel_Manual,Maruti,16.67,1
264,CAR_000265,Tata Sumo SE Plus BSIII,2002,100000.0,120000.0,Diesel,Individual,Manual,First Owner,265,23,Low,High,0.0,Diesel_Manual,Tata,0.83,1
265,CAR_000266,Mahindra Xylo E4,2010,160000.0,130000.0,Diesel,Individual,Manual,First Owner,266,15,Low,High,0.0,Diesel_Manual,Mahindra,1.23,1
266,CAR_000267,Maruti Swift ZDi BSIV,2017,450000.0,70000.0,Diesel,Individual,Manual,First Owner,267,8,Mid,Medium,0.0,Diesel_Manual,Maruti,6.43,1
267,CAR_000268,Maruti Ertiga VXI,2014,600000.0,20000.0,Petrol,Individual,Manual,First Owner,268,11,Mid,Low,0.0,Petrol_Manual,Maruti,30.0,0
268,CAR_000269,Hyundai i20 1.4 Magna ABS,2009,180000.0,60000.0,Diesel,Individual,Manual,Second Owner,269,16,Low,Medium,0.0,Diesel_Manual,Hyundai,3.0,1
269,CAR_000270,Maruti Alto LXi,2007,100000.0,195000.0,Petrol,Individual,Manual,First Owner,270,18,Low,Very High,0.0,Petrol_Manual,Maruti,0.51,0
270,CAR_000271,Maruti Alto LXi,2010,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,271,15,Low,High,0.0,Petrol_Manual,Maruti,1.33,0
271,CAR_000272,Hyundai Getz GL,2006,130000.0,80000.0,Petrol,Individual,Manual,Third Owner,272,19,Low,High,0.0,Petrol_Manual,Hyundai,1.62,0
272,CAR_000273,Maruti Wagon R LX BS IV,2011,160000.0,155000.0,Petrol,Individual,Manual,First Owner,273,14,Low,Very High,0.0,Petrol_Manual,Maruti,1.03,0
273,CAR_000274,Mahindra Bolero Power Plus SLX,2019,860000.0,35000.0,Diesel,Individual,Manual,First Owner,274,6,High,Medium,0.0,Diesel_Manual,Mahindra,24.57,1
274,CAR_000275,Hyundai i20 1.2 Spotz,2018,4461000.0,20000.0,Petrol,Individual,Manual,Second Owner,275,7,High,Low,0.0,Petrol_Manual,Hyundai,32.5,0
275,CAR_000276,Maruti Wagon R LXI Minor,2010,100000.0,80000.0,Petrol,Individual,Manual,Second Owner,276,15,Low,Medium,0.0,Petrol_Manual,Maruti,1.25,0
276,CAR_000277,Maruti Alto 800 VXI,2017,282000.0,20000.0,Petrol,Individual,Manual,First Owner,277,8,Low,Low,0.0,Petrol_Manual,Maruti,14.1,0
277,CAR_000278,Hyundai i20 Asta (o),2009,300000.0,50000.0,Petrol,Individual,Manual,First Owner,278,16,Low,Medium,0.0,Petrol_Manual,Hyundai,6.0,0
278,CAR_000279,Hyundai i20 Asta,2010,270000.0,110000.0,Petrol,Individual,Manual,Second Owner,279,15,Low,High,0.0,Petrol_Manual,Hyundai,2.45,0
279,CAR_000280,Hyundai Verna 1.6 SX CRDi (O),2013,550000.0,29000.0,Diesel,Individual,Manual,First Owner,280,12,Low,Low,0.0,Diesel_Manual,Hyundai,18.97,1
280,CAR_000281,Mahindra Xylo D2 BSIV,2013,450000.0,120000.0,Diesel,Individual,Manual,First Owner,281,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.75,1
281,CAR_000282,Mahindra Bolero SLX 2WD BSIII,2011,430000.0,70000.0,Diesel,Individual,Manual,Second Owner,282,14,Mid,Medium,0.0,Diesel_Manual,Mahindra,6.14,1
282,CAR_000283,Maruti Ertiga VXI ABS,2013,350000.0,40000.0,Electric,Individual,Manual,First Owner,283,12,Mid,Medium,0.0,Petrol_Manual,Maruti,8.75,0
283,CAR_000284,Honda Brio 1.2 E MT,2017,350000.0,20000.0,Petrol,Individual,Manual,First Owner,284,8,Mid,Low,0.0,Petrol_Manual,Honda,17.5,0
284,CAR_000285,Toyota Etios Liva G,2013,229999.0,50000.0,Petrol,Individual,Manual,Second Owner,285,12,Low,Medium,0.0,Petrol_Manual,Toyota,4.6,0
285,CAR_000286,Maruti Alto 800 LXI,2013,110000.0,50000.0,Petrol,Individual,Manual,First Owner,286,12,Low,Medium,0.0,Petrol_Manual,Maruti,2.2,0
286,CAR_000287,Maruti Swift Dzire 1.2 Vxi BSIV,2012,395000.0,49000.0,Petrol,Individual,Manual,Second Owner,287,13,Low,Medium,0.0,Petrol_Manual,Maruti,8.06,0
287,CAR_000288,Mahindra Scorpio VLS AT 2.2 mHAWK,2011,4461000.0,70000.0,Diesel,Individual,Automatic,Second Owner,288,14,High,Medium,0.0,Diesel_Automatic,Mahindra,9.29,1
288,CAR_000289,Mahindra Bolero SLX,2011,350000.0,90000.0,Diesel,Individual,Manual,Second Owner,289,14,Mid,High,0.0,Diesel_Manual,Mahindra,3.89,1
289,CAR_000290,Mahindra XUV500 W11 Option AWD,2020,1400000.0,60000.0,Diesel,Dealer,Manual,First Owner,290,5,Premium,Medium,0.0,Diesel_Manual,Mahindra,56.0,1
290,CAR_000291,Tata Tiago 1.2 Revotron XZ,2018,4461000.0,4000.0,Diesel,Individual,Manual,First Owner,291,7,Mid,Low,0.0,Petrol_Manual,Tata,90.0,0
291,CAR_000292,Mahindra Marazzo M8 8Str,2018,1300000.0,10000.0,Diesel,Individual,Manual,First Owner,292,7,Premium,Low,0.0,Diesel_Manual,Mahindra,130.0,1
292,CAR_000293,Renault Duster 110PS Diesel RxL,2015,800000.0,15000.0,Diesel,Individual,Manual,First Owner,293,10,High,Low,0.0,Diesel_Manual,Renault,53.33,1
293,CAR_000294,Hyundai EON Magna Plus,2018,229999.0,30000.0,Petrol,Individual,Manual,First Owner,294,7,Low,Low,0.0,Petrol_Manual,Hyundai,7.67,0
294,CAR_000295,Maruti Wagon R VXI,2014,315000.0,41000.0,Petrol,Dealer,Manual,First Owner,295,11,Mid,Medium,0.0,Petrol_Manual,Maruti,7.68,0
295,CAR_000296,Chevrolet Spark 1.0 LT,2012,125000.0,28000.0,Petrol,Individual,Manual,First Owner,296,13,Low,Low,0.0,Petrol_Manual,Chevrolet,4.46,0
296,CAR_000297,Maruti Baleno Alpha,2019,4461000.0,16000.0,Petrol,Individual,Manual,First Owner,297,6,High,Low,0.0,Petrol_Manual,Maruti,43.75,0
297,CAR_000298,Hyundai EON 1.0 Kappa Magna Plus,2015,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,298,10,Low,Medium,0.0,Petrol_Manual,Hyundai,2.29,0
298,CAR_000299,Ford Figo Aspire 1.2 Ti-VCT Trend,2017,4461000.0,10832.0,Petrol,Dealer,Manual,First Owner,299,8,Mid,Low,0.0,Petrol_Manual,Ford,48.93,0
299,CAR_000300,Ford Figo Aspire Facelift,2018,624000.0,14681.0,Diesel,Dealer,Manual,First Owner,300,7,High,Low,0.0,Diesel_Manual,Ford,42.5,1
300,CAR_000301,Hyundai i10 Era,2009,250000.0,50000.0,Petrol,Individual,Manual,First Owner,301,16,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
301,CAR_000302,Mahindra Bolero SLX,2016,585000.0,51000.0,Diesel,Dealer,Manual,First Owner,302,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,11.47,1
302,CAR_000303,Hyundai Verna 1.4 VTVT,2015,500000.0,60000.0,Petrol,Individual,Manual,First Owner,303,10,Mid,Low,0.0,Petrol_Manual,Hyundai,20.0,0
303,CAR_000304,Chevrolet Sail 1.2 LT ABS,2014,350000.0,30000.0,Petrol,Individual,Manual,First Owner,304,11,Mid,Low,0.0,Petrol_Manual,Chevrolet,11.67,0
304,CAR_000305,Honda City i-DTEC SV,2014,495000.0,60000.0,Diesel,Dealer,Manual,First Owner,305,11,Low,High,0.0,Diesel_Manual,Honda,6.19,1
305,CAR_000306,Hyundai Grand i10 1.2 Kappa Asta,2017,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,306,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,10.0,0
306,CAR_000307,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,430000.0,200000.0,Diesel,Individual,Manual,First Owner,307,10,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.15,1
307,CAR_000308,Hyundai Grand i10 1.2 Kappa Sportz Option,2013,345000.0,44000.0,Petrol,Dealer,Manual,First Owner,308,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.84,0
308,CAR_000309,Maruti Baleno Sigma 1.2,2015,4461000.0,10000.0,Petrol,Individual,Manual,First Owner,309,10,Mid,Low,0.0,Petrol_Manual,Maruti,50.0,0
309,CAR_000310,Tata Tiago 1.2 Revotron XZ,2019,4461000.0,19600.0,Petrol,Individual,Manual,First Owner,310,6,Low,Medium,0.0,Petrol_Manual,Tata,21.68,0
310,CAR_000311,Maruti Wagon R LXI,2005,80000.0,40000.0,Petrol,Individual,Manual,Second Owner,311,20,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
311,CAR_000312,Chevrolet Beat Diesel LS,2011,106000.0,60000.0,CNG,Individual,Manual,First Owner,312,14,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.12,1
312,CAR_000313,Maruti A-Star Vxi,2009,180000.0,46730.0,Petrol,Individual,Manual,First Owner,313,16,Low,Medium,0.0,Petrol_Manual,Maruti,3.85,0
313,CAR_000314,Datsun GO A,2015,250000.0,40000.0,Petrol,Individual,Manual,Second Owner,314,10,Low,Medium,0.0,Petrol_Manual,Datsun,6.25,0
314,CAR_000315,Skoda Yeti Ambition 4WD,2011,4461000.0,80000.0,LPG,Individual,Manual,First Owner,315,14,Mid,High,0.0,Diesel_Manual,Skoda,4.38,1
315,CAR_000316,Maruti Zen LXi BSII,2002,4461000.0,60000.0,Petrol,Individual,Manual,Second Owner,316,23,Low,High,0.0,Petrol_Manual,Maruti,0.75,0
316,CAR_000317,Hyundai Verna CRDi 1.6 EX,2018,900000.0,60000.0,Diesel,Individual,Manual,First Owner,317,7,High,Medium,0.0,Diesel_Manual,Hyundai,37.5,1
317,CAR_000318,Ford Aspire Titanium Diesel BSIV,2017,700000.0,21170.0,Diesel,Dealer,Manual,First Owner,318,8,High,Low,0.0,Diesel_Manual,Ford,33.07,1
318,CAR_000319,BMW 3 Series 320d Sport,2013,4461000.0,60000.0,Diesel,Dealer,Automatic,First Owner,319,12,Premium,Very High,0.0,Diesel_Automatic,BMW,11.96,1
319,CAR_000320,Maruti Ertiga SHVS ZDI Plus,2017,1000000.0,40000.0,Diesel,Individual,Manual,First Owner,320,8,High,Medium,0.0,Diesel_Manual,Maruti,25.0,1
320,CAR_000321,BMW X1 sDrive 20d Exclusive,2011,1375000.0,60000.0,Diesel,Individual,Automatic,First Owner,321,14,Premium,Medium,0.0,Diesel_Automatic,BMW,22.92,1
321,CAR_000322,Toyota Innova Crysta 2.4 VX MT,2016,1800000.0,60000.0,Diesel,Dealer,Manual,First Owner,322,9,Premium,High,0.0,Diesel_Manual,Toyota,12.73,1
322,CAR_000323,Chevrolet Tavera Neo 3 10 Seats BSIV,2007,160000.0,120000.0,Diesel,Individual,Manual,Third Owner,323,18,Low,High,0.0,Diesel_Manual,Chevrolet,1.33,1
323,CAR_000324,Mitsubishi Outlander 2.4,2012,525000.0,140000.0,Petrol,Individual,Automatic,First Owner,324,13,Mid,High,0.0,Petrol_Automatic,Mitsubishi,3.75,0
324,CAR_000325,Mahindra XUV500 W8 2WD,2012,850000.0,212814.0,Diesel,Dealer,Manual,First Owner,325,13,High,Medium,0.0,Diesel_Manual,Mahindra,3.99,1
325,CAR_000326,Maruti Baleno Alpha 1.3,2019,800000.0,30000.0,Diesel,Individual,Manual,First Owner,326,6,High,Low,0.0,Diesel_Manual,Maruti,26.67,1
326,CAR_000327,Ford Figo Aspire 1.5 TDCi Titanium,2017,700000.0,88635.0,Diesel,Dealer,Manual,First Owner,327,8,High,High,0.0,Diesel_Manual,Ford,7.9,1
327,CAR_000328,Maruti Swift Dzire VDI,2018,800000.0,25000.0,Diesel,Individual,Manual,First Owner,328,7,High,Low,0.0,Diesel_Manual,Maruti,32.0,1
328,CAR_000329,Hyundai i20 1.2 Spotz,2017,575000.0,20000.0,Petrol,Individual,Manual,Second Owner,329,8,Mid,Low,0.0,Petrol_Manual,Hyundai,28.75,0
329,CAR_000330,Renault Duster 110PS Diesel RxZ,2012,350000.0,110000.0,Diesel,Individual,Manual,Second Owner,330,13,Low,Medium,0.0,Diesel_Manual,Renault,3.18,1
330,CAR_000331,Maruti Swift Vdi BSIII,2008,250000.0,120000.0,Diesel,Individual,Manual,Third Owner,331,17,Low,High,0.0,Diesel_Manual,Maruti,2.08,1
331,CAR_000332,Ford Endeavour Titanium 4X2,2011,600000.0,149674.0,Diesel,Dealer,Manual,Second Owner,332,14,Low,High,0.0,Diesel_Manual,Ford,4.01,1
332,CAR_000333,Mahindra Scorpio 2.6 Turbo 9 Str,2007,370000.0,100000.0,Diesel,Individual,Manual,Third Owner,333,18,Mid,High,0.0,Diesel_Manual,Mahindra,3.7,1
333,CAR_000334,Maruti Baleno Vxi,2006,80000.0,60000.0,Petrol,Individual,Manual,Third Owner,334,19,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
334,CAR_000335,Tata Indica DLX,2003,50000.0,120000.0,Diesel,Individual,Manual,Second Owner,335,22,Low,High,0.0,Diesel_Manual,Tata,0.42,1
335,CAR_000336,Hyundai Verna CRDi 1.6 EX,2018,900000.0,25000.0,Diesel,Individual,Manual,First Owner,336,7,High,Low,0.0,Diesel_Manual,Hyundai,36.0,1
336,CAR_000337,Maruti Wagon R VXI BS IV,2014,300000.0,35000.0,Petrol,Individual,Manual,Second Owner,337,11,Low,Medium,0.0,Petrol_Manual,Maruti,8.57,0
337,CAR_000338,Tata Indica LSI,2002,125000.0,40000.0,Petrol,Individual,Manual,First Owner,338,23,Low,Medium,0.0,Petrol_Manual,Tata,3.12,0
338,CAR_000339,Maruti Swift Dzire VDI,2015,490000.0,60000.0,Diesel,Individual,Manual,Second Owner,339,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.17,1
339,CAR_000340,Toyota Etios Liva GD,2012,300000.0,50000.0,Diesel,Individual,Manual,Second Owner,340,13,Low,Medium,0.0,Diesel_Manual,Toyota,6.0,1
340,CAR_000341,Tata Nano STD,2013,55000.0,25000.0,Petrol,Individual,Manual,Fourth & Above Owner,341,12,Low,Low,0.0,Petrol_Manual,Tata,2.2,0
341,CAR_000342,Hyundai Grand i10 Sportz,2016,350000.0,25000.0,Petrol,Individual,Manual,First Owner,342,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,14.0,0
342,CAR_000343,Hyundai Grand i10 1.2 CRDi Asta,2019,755000.0,8000.0,Diesel,Individual,Manual,First Owner,343,6,High,Low,0.0,Diesel_Manual,Hyundai,94.38,1
343,CAR_000344,Maruti Ertiga SHVS VDI,2017,720000.0,60000.0,Diesel,Individual,Manual,First Owner,344,8,High,Medium,0.0,Diesel_Manual,Maruti,12.0,1
344,CAR_000345,Mahindra Bolero SLX,2006,229999.0,200000.0,Diesel,Individual,Manual,Third Owner,345,19,Low,Very High,0.0,Diesel_Manual,Mahindra,1.15,1
345,CAR_000346,Mahindra Scorpio S7 140 BSIV,2018,1100000.0,20000.0,Electric,Individual,Manual,First Owner,346,7,Premium,Low,0.0,Diesel_Manual,Mahindra,55.0,1
346,CAR_000347,Volkswagen Polo Petrol Comfortline 1.2L,2013,400000.0,60000.0,Petrol,Individual,Manual,First Owner,347,12,Mid,High,0.0,Petrol_Manual,Volkswagen,4.0,0
347,CAR_000348,Hyundai Xcent 1.1 CRDi Base,2015,350000.0,70000.0,Diesel,Individual,Manual,Second Owner,348,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,5.0,1
348,CAR_000349,Chevrolet Beat Diesel PS,2012,159000.0,68000.0,Diesel,Dealer,Manual,First Owner,349,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.34,1
349,CAR_000350,Maruti Swift ZXi BSIV,2017,550000.0,38000.0,Petrol,Dealer,Manual,First Owner,350,8,Mid,Medium,0.0,Petrol_Manual,Maruti,14.47,0
350,CAR_000351,Maruti Ritz VDi,2013,335000.0,80000.0,Diesel,Dealer,Manual,First Owner,351,12,Mid,High,0.0,Diesel_Manual,Maruti,4.19,1
351,CAR_000352,Tata Indigo CS Emax CNG GLX,2014,4461000.0,72000.0,Petrol,Dealer,Manual,First Owner,352,11,Low,High,0.0,CNG_Manual,Tata,2.57,0
352,CAR_000353,Maruti Eeco 5 STR With AC Plus HTR CNG,2012,229999.0,75000.0,CNG,Dealer,Manual,First Owner,353,13,Low,High,0.0,CNG_Manual,Maruti,3.07,0
353,CAR_000354,Maruti Estilo LXI,2014,225000.0,60000.0,Petrol,Dealer,Manual,First Owner,354,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.23,0
354,CAR_000355,Maruti Swift VXI with ABS,2007,250000.0,120000.0,Diesel,Individual,Manual,Second Owner,355,18,Low,High,0.0,Petrol_Manual,Maruti,2.08,0
355,CAR_000356,Tata Zest Quadrajet 1.3 XM,2015,400000.0,100000.0,Diesel,Individual,Manual,First Owner,356,10,Mid,High,0.0,Diesel_Manual,Tata,4.0,1
356,CAR_000357,Maruti Swift VDI BSIV,2014,470000.0,163000.0,Diesel,Individual,Manual,First Owner,357,11,Mid,Very High,0.0,Diesel_Manual,Maruti,2.88,1
357,CAR_000358,Maruti Alto K10 VXI AGS Optional,2018,300000.0,25000.0,CNG,Individual,Automatic,First Owner,358,7,Low,Low,0.0,Petrol_Automatic,Maruti,12.0,0
358,CAR_000359,Maruti Alto LX,2011,145000.0,25000.0,Petrol,Individual,Manual,First Owner,359,14,Low,Medium,0.0,Petrol_Manual,Maruti,5.8,0
359,CAR_000360,Maruti SX4 Vxi BSIII,2007,100000.0,90000.0,Petrol,Individual,Manual,Second Owner,360,18,Low,High,0.0,Petrol_Manual,Maruti,1.11,0
360,CAR_000361,Mahindra Scorpio SLE BSIV,2014,500000.0,60000.0,LPG,Individual,Manual,Second Owner,361,11,Mid,Medium,0.0,Diesel_Manual,Mahindra,7.14,1
361,CAR_000362,Hyundai Santro Xing GLS CNG,2010,130000.0,98000.0,CNG,Individual,Manual,Second Owner,362,15,Low,High,0.0,CNG_Manual,Hyundai,1.33,0
362,CAR_000363,Chevrolet Cruze LTZ,2010,240000.0,81925.0,Diesel,Individual,Manual,First Owner,363,15,Low,High,0.0,Diesel_Manual,Chevrolet,2.93,1
363,CAR_000364,Ford Figo Diesel Celebration Edition,2013,150000.0,80000.0,Diesel,Individual,Manual,Second Owner,364,12,Low,High,0.0,Diesel_Manual,Ford,1.88,1
364,CAR_000365,Renault KWID RXT Optional,2017,270000.0,40000.0,Petrol,Individual,Manual,First Owner,365,8,Low,Medium,0.0,Petrol_Manual,Renault,6.75,0
365,CAR_000366,Maruti Swift Dzire VDI,2014,400000.0,90000.0,Electric,Individual,Manual,First Owner,366,11,Mid,High,0.0,Diesel_Manual,Maruti,4.44,1
366,CAR_000367,Hyundai EON Era Plus,2012,4461000.0,46000.0,Petrol,Individual,Manual,First Owner,367,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.35,0
367,CAR_000368,Mahindra XUV500 W6 2WD,2012,550000.0,80000.0,Diesel,Individual,Manual,First Owner,368,13,Mid,High,0.0,Diesel_Manual,Mahindra,6.88,1
368,CAR_000369,Renault KWID 1.0 RXT Optional,2018,325000.0,35000.0,Petrol,Individual,Manual,First Owner,369,7,Mid,Medium,0.0,Petrol_Manual,Renault,9.29,0
369,CAR_000370,Skoda Rapid 1.5 TDI AT Style BSIV,2017,860000.0,82080.0,Petrol,Individual,Automatic,First Owner,370,8,High,High,0.0,Diesel_Automatic,Skoda,10.48,1
370,CAR_000371,Mahindra XUV500 W8 2WD,2012,600000.0,200000.0,Diesel,Individual,Manual,First Owner,371,13,Mid,Very High,0.0,Diesel_Manual,Mahindra,3.0,1
371,CAR_000372,Chevrolet Spark 1.0,2009,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,372,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.79,0
372,CAR_000373,Maruti 800 AC,2000,60000.0,40000.0,Petrol,Individual,Manual,Third Owner,373,25,Low,Medium,0.0,Petrol_Manual,Maruti,1.5,0
373,CAR_000374,Chevrolet Tavera Neo 3 10 Seats BSIV,2010,300000.0,120000.0,Diesel,Individual,Manual,Third Owner,374,15,Low,High,0.0,Diesel_Manual,Chevrolet,2.5,1
374,CAR_000375,Ford Fiesta 1.6 ZXi Duratec,2007,190000.0,60000.0,Petrol,Individual,Manual,Third Owner,375,18,Low,Medium,0.0,Petrol_Manual,Ford,3.17,0
375,CAR_000376,Maruti Ertiga VDI,2013,400000.0,110000.0,Diesel,Individual,Manual,First Owner,376,12,Mid,High,0.0,Diesel_Manual,Maruti,3.64,1
376,CAR_000377,Tata Manza Aura Quadrajet BS IV,2011,4461000.0,97000.0,Diesel,Individual,Manual,Second Owner,377,14,Low,High,0.0,Diesel_Manual,Tata,2.68,1
377,CAR_000378,Maruti Swift VVT VXI,2017,400000.0,30000.0,Petrol,Individual,Manual,First Owner,378,8,Mid,Low,0.0,Petrol_Manual,Maruti,13.33,0
378,CAR_000379,Hyundai i20 Sportz Petrol,2010,250000.0,40000.0,Petrol,Individual,Manual,First Owner,379,15,Low,Medium,0.0,Petrol_Manual,Hyundai,6.25,0
379,CAR_000380,Hyundai Grand i10 1.2 Kappa Asta,2019,600000.0,10000.0,Petrol,Individual,Manual,First Owner,380,6,Mid,Low,0.0,Petrol_Manual,Hyundai,60.0,0
380,CAR_000381,Maruti Zen Estilo VXI BSIV,2010,175000.0,52047.0,Petrol,Individual,Manual,First Owner,381,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.36,0
381,CAR_000382,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2011,450000.0,200000.0,Diesel,Individual,Manual,Third Owner,382,14,Mid,Very High,0.0,Diesel_Manual,Toyota,2.25,1
382,CAR_000383,Tata Tiago 1.2 Revotron XZA,2019,350000.0,15000.0,Petrol,Individual,Automatic,Second Owner,383,6,Mid,Low,0.0,Petrol_Automatic,Tata,23.33,0
383,CAR_000384,Maruti Vitara Brezza ZDi,2018,800000.0,50000.0,Diesel,Individual,Manual,First Owner,384,7,High,Medium,0.0,Diesel_Manual,Maruti,16.0,1
384,CAR_000385,Maruti Swift VXI BSIII,2011,250000.0,60000.0,Petrol,Individual,Manual,Second Owner,385,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.57,0
385,CAR_000386,Maruti Swift VXI BSIV,2019,600000.0,60000.0,Diesel,Individual,Manual,First Owner,386,6,Low,Low,0.0,Petrol_Manual,Maruti,60.0,0
386,CAR_000387,Maruti Swift VDI BSIV,2017,600000.0,90000.0,Diesel,Individual,Manual,First Owner,387,8,Mid,High,0.0,Diesel_Manual,Maruti,6.67,1
387,CAR_000388,Maruti Alto 800 LXI,2016,130000.0,40000.0,Petrol,Individual,Manual,First Owner,388,9,Low,Medium,0.0,Petrol_Manual,Maruti,3.25,0
388,CAR_000389,Maruti Wagon R LXI,2006,90000.0,62009.0,Petrol,Individual,Manual,Second Owner,389,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.45,0
389,CAR_000390,Hyundai i20 Active 1.2 S,2017,595000.0,33100.0,Petrol,Individual,Manual,Second Owner,390,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,17.98,0
390,CAR_000391,Hyundai EON Era Plus,2014,190000.0,40000.0,Petrol,Individual,Manual,Second Owner,391,11,Low,Medium,0.0,Petrol_Manual,Hyundai,4.75,0
391,CAR_000392,Hyundai Santro GLS I - Euro I,1999,50000.0,120000.0,Petrol,Individual,Manual,Second Owner,392,26,Low,High,0.0,Petrol_Manual,Hyundai,0.42,0
392,CAR_000393,Maruti Wagon R LXI Minor,2007,95000.0,80000.0,Petrol,Individual,Manual,First Owner,393,18,Low,High,0.0,Petrol_Manual,Maruti,1.19,0
393,CAR_000394,Tata Safari Storme VX,2016,1000000.0,70000.0,Diesel,Individual,Manual,First Owner,394,9,High,Medium,0.0,Diesel_Manual,Tata,14.29,1
394,CAR_000395,Mahindra Scorpio REV 116,2006,220000.0,220000.0,Petrol,Individual,Manual,Second Owner,395,19,Low,Very High,0.0,Petrol_Manual,Mahindra,1.0,0
395,CAR_000396,Mahindra Bolero SLX,2010,400000.0,50000.0,Diesel,Individual,Manual,First Owner,396,15,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.0,1
396,CAR_000397,Datsun GO Plus A,2015,315000.0,45000.0,CNG,Individual,Manual,Second Owner,397,10,Mid,Medium,0.0,Petrol_Manual,Datsun,7.0,0
397,CAR_000398,Toyota Fortuner 4x4 MT,2014,1000000.0,110000.0,Diesel,Individual,Manual,First Owner,398,11,High,High,0.0,Diesel_Manual,Toyota,9.09,1
398,CAR_000399,Ford Ecosport 1.5 DV5 MT Titanium,2014,500000.0,49000.0,Diesel,Individual,Manual,First Owner,399,11,Mid,Medium,0.0,Diesel_Manual,Ford,10.2,1
399,CAR_000400,Mahindra XUV500 W11 AT BSIV,2018,1600000.0,60000.0,Diesel,Individual,Automatic,First Owner,400,7,Premium,Medium,0.0,Diesel_Automatic,Mahindra,45.71,1
400,CAR_000401,Mahindra Supro VX 8 Str,2017,500000.0,60000.0,Diesel,Individual,Manual,First Owner,401,8,Mid,Very High,0.0,Diesel_Manual,Mahindra,3.12,1
401,CAR_000402,Maruti Alto LX,2007,70000.0,180000.0,Petrol,Individual,Manual,Third Owner,402,18,Low,Very High,0.0,Petrol_Manual,Maruti,0.39,0
402,CAR_000403,Maruti 800 AC,2007,105000.0,60000.0,Petrol,Individual,Manual,Second Owner,403,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.75,0
403,CAR_000404,Ford Endeavour 2.5L 4X2 MT,2008,400000.0,110000.0,Diesel,Individual,Manual,Third Owner,404,17,Mid,High,0.0,Diesel_Manual,Ford,3.64,1
404,CAR_000405,Maruti Wagon R LXI,2004,80000.0,60000.0,Petrol,Individual,Manual,Third Owner,405,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.33,0
405,CAR_000406,Maruti Alto 800 VXI,2016,300000.0,30000.0,Petrol,Individual,Manual,First Owner,406,9,Low,Low,0.0,Petrol_Manual,Maruti,10.0,0
406,CAR_000407,Mahindra KUV 100 D75 K2,2017,409999.0,40000.0,Diesel,Individual,Manual,First Owner,407,8,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.25,1
407,CAR_000408,Maruti Swift Glam,2009,250000.0,70000.0,Petrol,Individual,Manual,First Owner,408,16,Low,Medium,0.0,Petrol_Manual,Maruti,3.57,0
408,CAR_000409,Tata New Safari DICOR 2.2 EX 4x2,2010,250000.0,120000.0,Diesel,Individual,Manual,Second Owner,409,15,Low,High,0.0,Diesel_Manual,Tata,2.08,1
409,CAR_000410,Maruti Swift Dzire VDI,2012,215000.0,80000.0,Diesel,Individual,Manual,Third Owner,410,13,Low,High,0.0,Diesel_Manual,Maruti,2.69,1
410,CAR_000411,Toyota Fortuner 4x4 MT,2015,1400000.0,120000.0,Diesel,Individual,Manual,First Owner,411,10,Low,Medium,0.0,Diesel_Manual,Toyota,11.67,1
411,CAR_000412,Honda Brio 1.2 VX MT,2018,475000.0,20000.0,Petrol,Individual,Manual,First Owner,412,7,Mid,Low,0.0,Petrol_Manual,Honda,23.75,0
412,CAR_000413,Honda Brio 1.2 VX MT,2018,475000.0,22000.0,Petrol,Individual,Manual,First Owner,413,7,Mid,Low,0.0,Petrol_Manual,Honda,21.59,0
413,CAR_000414,Chevrolet Spark 1.0 LS,2008,75000.0,35000.0,Petrol,Individual,Manual,Second Owner,414,17,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.14,0
414,CAR_000415,Chevrolet Sail Hatchback LS ABS,2013,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,415,12,Low,High,0.0,Diesel_Manual,Chevrolet,1.67,1
415,CAR_000416,Tata Tiago NRG Petrol,2019,575000.0,8000.0,Petrol,Individual,Manual,First Owner,416,6,Mid,Medium,0.0,Petrol_Manual,Tata,71.88,0
416,CAR_000417,Maruti Ritz VDi,2011,150000.0,40000.0,Diesel,Individual,Manual,Second Owner,417,14,Low,Medium,0.0,Diesel_Manual,Maruti,3.75,1
417,CAR_000418,Maruti Vitara Brezza ZDi Plus,2018,850000.0,10000.0,Diesel,Individual,Manual,First Owner,418,7,High,Low,0.0,Diesel_Manual,Maruti,85.0,1
418,CAR_000419,Hyundai Santro Era,2019,350000.0,30000.0,Petrol,Individual,Manual,First Owner,419,6,Mid,Low,0.0,Petrol_Manual,Hyundai,11.67,0
419,CAR_000420,Ford Endeavour 2.5L 4X2 MT,2011,550000.0,54000.0,Diesel,Individual,Manual,Second Owner,420,14,Mid,Medium,0.0,Diesel_Manual,Ford,10.19,1
420,CAR_000421,Hyundai Verna 1.4 VTVT,2017,330000.0,80577.0,Petrol,Individual,Manual,First Owner,421,8,Mid,High,0.0,Petrol_Manual,Hyundai,4.1,0
421,CAR_000422,Honda City i-DTEC V,2017,1044999.0,25000.0,Diesel,Individual,Manual,First Owner,422,8,Premium,Low,0.0,Diesel_Manual,Honda,41.8,1
422,CAR_000423,Hyundai Verna 1.6 CRDI,2011,300000.0,127500.0,Diesel,Individual,Manual,Second Owner,423,14,Low,Medium,0.0,Diesel_Manual,Hyundai,2.35,1
423,CAR_000424,Hyundai EON D Lite Plus,2015,235000.0,40903.0,Petrol,Dealer,Manual,First Owner,424,10,Low,Medium,0.0,Petrol_Manual,Hyundai,5.75,0
424,CAR_000425,Ford Figo Petrol Titanium,2011,120000.0,60000.0,Petrol,Individual,Manual,Third Owner,425,14,Low,Medium,0.0,Petrol_Manual,Ford,2.0,0
425,CAR_000426,Maruti Wagon R VXI,2004,130000.0,46000.0,Petrol,Individual,Manual,First Owner,426,21,Low,Medium,0.0,Petrol_Manual,Maruti,2.83,0
426,CAR_000427,Ford Endeavour 4x4 XLT,2004,4461000.0,22288.0,Diesel,Dealer,Manual,Second Owner,427,21,Low,Medium,0.0,Diesel_Manual,Ford,12.34,1
427,CAR_000428,Volkswagen Jetta 2.0 TDI Trendline,2011,270000.0,180000.0,Diesel,Individual,Manual,Second Owner,428,14,Low,Very High,0.0,Diesel_Manual,Volkswagen,1.5,1
428,CAR_000429,Volkswagen Polo 1.5 TDI Highline,2015,500000.0,30000.0,Diesel,Individual,Manual,Second Owner,429,10,Mid,Medium,0.0,Diesel_Manual,Volkswagen,16.67,1
429,CAR_000430,Hyundai Grand i10 Sportz,2015,400000.0,60000.0,LPG,Dealer,Manual,First Owner,430,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.48,0
430,CAR_000431,Hyundai i20 Asta 1.2,2015,600000.0,64484.0,Petrol,Dealer,Manual,First Owner,431,10,Low,Medium,0.0,Petrol_Manual,Hyundai,9.3,0
431,CAR_000432,Audi Q5 2.0 TDI,2012,1350000.0,90000.0,Diesel,Individual,Manual,Second Owner,432,13,Premium,High,0.0,Diesel_Automatic,Audi,15.0,1
432,CAR_000433,Renault Duster 85PS Diesel RxE,2013,4461000.0,75976.0,Diesel,Dealer,Manual,Second Owner,433,12,Mid,High,0.0,Diesel_Manual,Renault,5.26,1
433,CAR_000434,Ford Figo Diesel Titanium,2011,220000.0,70000.0,Diesel,Individual,Manual,Second Owner,434,14,Low,Medium,0.0,Diesel_Manual,Ford,3.14,1
434,CAR_000435,Honda Brio E MT,2012,300000.0,85962.0,Petrol,Dealer,Manual,Second Owner,435,13,Low,High,0.0,Petrol_Manual,Honda,3.49,0
435,CAR_000436,Maruti Omni E MPI STD BS IV,2018,250000.0,120000.0,Petrol,Individual,Manual,First Owner,436,7,Low,High,0.0,Petrol_Manual,Maruti,2.08,0
436,CAR_000437,Honda Mobilio V i DTEC,2015,600000.0,60000.0,Electric,Individual,Manual,First Owner,437,10,Mid,High,0.0,Diesel_Manual,Honda,5.0,1
437,CAR_000438,Honda Brio S Option AT,2012,285000.0,75000.0,Petrol,Individual,Automatic,Second Owner,438,13,Low,High,0.0,Petrol_Automatic,Honda,3.8,0
438,CAR_000439,Honda Accord 2.4 MT,2009,4461000.0,57035.0,Petrol,Dealer,Manual,First Owner,439,16,Low,Medium,0.0,Petrol_Manual,Honda,6.14,0
439,CAR_000440,Mahindra Bolero SLX,2011,170000.0,70000.0,Diesel,Individual,Manual,Third Owner,440,14,Low,Medium,0.0,Diesel_Manual,Mahindra,2.43,1
440,CAR_000441,Ford Figo 1.2P Ambiente MT,2015,229999.0,45000.0,Petrol,Individual,Manual,First Owner,441,10,Low,Medium,0.0,Petrol_Manual,Ford,5.11,0
441,CAR_000442,Mahindra XUV500 W6 2WD,2013,500000.0,72104.0,Diesel,Dealer,Manual,Second Owner,442,12,Mid,High,0.0,Diesel_Manual,Mahindra,6.93,1
442,CAR_000443,Tata Nano Lx,2011,75000.0,15000.0,Petrol,Individual,Manual,First Owner,443,14,Low,Low,0.0,Petrol_Manual,Tata,5.0,0
443,CAR_000444,Hyundai i20 Active 1.2 SX,2016,4461000.0,30000.0,Petrol,Individual,Manual,First Owner,444,9,Mid,Low,0.0,Petrol_Manual,Hyundai,20.0,0
444,CAR_000445,Tata Indigo LS Dicor,2011,220000.0,80000.0,Diesel,Individual,Manual,First Owner,445,14,Low,High,0.0,Diesel_Manual,Tata,2.75,1
445,CAR_000446,Tata Indica Vista Terra 1.4 TDI,2009,150000.0,164000.0,Diesel,Individual,Manual,Second Owner,446,16,Low,Very High,0.0,Diesel_Manual,Tata,0.91,1
446,CAR_000447,Maruti Alto LX,2006,65000.0,124439.0,Petrol,Individual,Manual,Second Owner,447,19,Low,High,0.0,Petrol_Manual,Maruti,0.52,0
447,CAR_000448,Maruti Zen VXI,2006,120000.0,77000.0,Petrol,Individual,Manual,Second Owner,448,19,Low,High,0.0,Petrol_Manual,Maruti,1.56,0
448,CAR_000449,Maruti Swift Dzire VDI,2014,240000.0,90000.0,Diesel,Individual,Manual,First Owner,449,11,Low,High,0.0,Diesel_Manual,Maruti,2.67,1
449,CAR_000450,Renault KWID 1.0 RXT Optional,2017,390000.0,30000.0,Petrol,Individual,Manual,First Owner,450,8,Mid,Low,0.0,Petrol_Manual,Renault,13.0,0
450,CAR_000451,Chevrolet Beat LS,2012,180000.0,50000.0,Petrol,Individual,Manual,First Owner,451,13,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.6,0
451,CAR_000452,Maruti Swift VDI,2012,320000.0,80000.0,Diesel,Individual,Manual,Second Owner,452,13,Mid,High,0.0,Diesel_Manual,Maruti,4.0,1
452,CAR_000453,Maruti Alto 800 LXI,2018,310000.0,5000.0,Petrol,Individual,Manual,First Owner,453,7,Mid,Low,0.0,Petrol_Manual,Maruti,62.0,0
453,CAR_000454,Maruti Wagon R AMT VXI,2018,420000.0,60000.0,Petrol,Individual,Automatic,First Owner,454,7,Mid,Medium,0.0,Petrol_Automatic,Maruti,17.5,0
454,CAR_000455,Tata Tiago NRG Petrol,2019,550000.0,10000.0,Petrol,Individual,Manual,First Owner,455,6,Mid,Low,0.0,Petrol_Manual,Tata,55.0,0
455,CAR_000456,Maruti Ertiga SHVS VDI,2016,550000.0,78000.0,Diesel,Individual,Manual,Third Owner,456,9,Mid,High,0.0,Diesel_Manual,Maruti,7.05,1
456,CAR_000457,Mahindra TUV 300 T4 Plus,2018,760000.0,50000.0,Diesel,Individual,Manual,First Owner,457,7,High,Medium,0.0,Diesel_Manual,Mahindra,15.2,1
457,CAR_000458,Maruti 800 AC BSII,2001,4461000.0,100000.0,Petrol,Individual,Manual,Second Owner,458,24,Low,High,0.0,Petrol_Manual,Maruti,0.43,0
458,CAR_000459,Maruti Omni MPI STD BSIV,2018,250000.0,20000.0,Petrol,Individual,Manual,First Owner,459,7,Low,Low,0.0,Petrol_Manual,Maruti,12.5,0
459,CAR_000460,Hyundai Santro Magna BSIV,2019,500000.0,1250.0,Petrol,Individual,Manual,First Owner,460,6,Mid,Low,0.0,Petrol_Manual,Hyundai,400.0,0
460,CAR_000461,Tata Zest Revotron 1.2 XT,2015,380000.0,17152.0,Petrol,Individual,Manual,First Owner,461,10,Mid,Low,0.0,Petrol_Manual,Tata,22.15,0
461,CAR_000462,Mahindra XUV500 W11 Option AT AWD,2019,4461000.0,5000.0,Diesel,Individual,Automatic,First Owner,462,6,Premium,Low,0.0,Diesel_Automatic,Mahindra,370.0,1
462,CAR_000463,Volkswagen Polo Diesel Trendline 1.2L,2012,350000.0,77000.0,Diesel,Individual,Manual,First Owner,463,13,Mid,High,0.0,Diesel_Manual,Volkswagen,4.55,1
463,CAR_000464,Hyundai Verna CRDi,2008,4461000.0,90000.0,Diesel,Individual,Manual,Second Owner,464,17,Low,High,0.0,Diesel_Manual,Hyundai,2.17,1
464,CAR_000465,Chevrolet Beat LS,2011,225000.0,80000.0,Petrol,Individual,Manual,Second Owner,465,14,Low,High,0.0,Petrol_Manual,Chevrolet,2.81,0
465,CAR_000466,Toyota Innova 2.5 G (Diesel) 7 Seater,2015,1125000.0,65000.0,Diesel,Individual,Manual,First Owner,466,10,Premium,Medium,0.0,Diesel_Manual,Toyota,17.31,1
466,CAR_000467,Tata Indica Vista Quadrajet LX,2011,133000.0,60000.0,Diesel,Individual,Manual,First Owner,467,14,Low,High,0.0,Diesel_Manual,Tata,1.66,1
467,CAR_000468,Hyundai EON Era Plus,2013,200000.0,24005.0,Petrol,Individual,Manual,First Owner,468,12,Low,Low,0.0,Petrol_Manual,Hyundai,8.33,0
468,CAR_000469,Maruti Vitara Brezza ZDi Plus,2018,4461000.0,20000.0,Diesel,Individual,Manual,First Owner,469,7,High,Low,0.0,Diesel_Manual,Maruti,47.0,1
469,CAR_000470,Renault Duster 85PS Diesel RxL,2014,400000.0,120000.0,Diesel,Individual,Manual,First Owner,470,11,Mid,High,0.0,Diesel_Manual,Renault,3.33,1
470,CAR_000471,Hyundai i10 Era 1.1,2009,120000.0,120000.0,Petrol,Individual,Manual,Second Owner,471,16,Low,High,0.0,Petrol_Manual,Hyundai,1.0,0
471,CAR_000472,Maruti Swift Dzire ZDI,2015,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,472,10,High,High,0.0,Diesel_Manual,Maruti,7.81,1
472,CAR_000473,Maruti Swift 1.3 VXi,2005,110000.0,80000.0,Petrol,Individual,Manual,First Owner,473,20,Low,High,0.0,Petrol_Manual,Maruti,1.38,0
473,CAR_000474,Renault Duster 110PS Diesel RxZ,2012,490000.0,100000.0,Diesel,Individual,Manual,First Owner,474,13,Mid,High,0.0,Diesel_Manual,Renault,4.9,1
474,CAR_000475,Maruti Alto LX,2009,140000.0,60000.0,Petrol,Individual,Manual,Second Owner,475,16,Low,High,0.0,Petrol_Manual,Maruti,1.17,0
475,CAR_000476,Ford Figo Diesel ZXI,2011,175000.0,149000.0,Diesel,Individual,Manual,Second Owner,476,14,Low,High,0.0,Diesel_Manual,Ford,1.17,1
476,CAR_000477,Maruti Wagon R LXI BSIII,2005,90000.0,65000.0,Petrol,Individual,Manual,Second Owner,477,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.38,0
477,CAR_000478,Maruti Swift Ldi BSIV,2009,275000.0,100000.0,Diesel,Individual,Manual,Second Owner,478,16,Low,High,0.0,Diesel_Manual,Maruti,2.75,1
478,CAR_000479,Volkswagen Vento Diesel Highline,2011,4461000.0,65000.0,Diesel,Individual,Manual,Second Owner,479,14,Low,Medium,0.0,Diesel_Manual,Volkswagen,4.62,1
479,CAR_000480,Datsun GO Plus T Option BSIV,2018,270000.0,19000.0,Petrol,Individual,Manual,Second Owner,480,7,Low,Low,0.0,Petrol_Manual,Datsun,14.21,0
480,CAR_000481,Chevrolet Beat LT,2010,130000.0,80000.0,Petrol,Individual,Manual,Second Owner,481,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.62,0
481,CAR_000482,Maruti Alto K10 VXI Airbag,2015,275000.0,30000.0,Petrol,Individual,Manual,First Owner,482,10,Low,Low,0.0,Petrol_Manual,Maruti,9.17,0
482,CAR_000483,Maruti Swift VDI,2012,352000.0,109000.0,Diesel,Individual,Manual,First Owner,483,13,Mid,High,0.0,Diesel_Manual,Maruti,3.23,1
483,CAR_000484,Hyundai EON LPG Magna Plus,2012,180000.0,90000.0,LPG,Individual,Manual,First Owner,484,13,Low,High,0.0,LPG_Manual,Hyundai,2.0,0
484,CAR_000485,Maruti Alto LX,2012,150000.0,90000.0,Petrol,Individual,Manual,First Owner,485,13,Low,High,0.0,Petrol_Manual,Maruti,1.67,0
485,CAR_000486,Volkswagen Vento Diesel Highline,2011,300000.0,65000.0,Diesel,Individual,Manual,Second Owner,486,14,Low,Medium,0.0,Diesel_Manual,Volkswagen,4.62,1
486,CAR_000487,Maruti Swift Vdi BSIII,2010,300000.0,110000.0,Diesel,Individual,Manual,Second Owner,487,15,Low,High,0.0,Diesel_Manual,Maruti,2.73,1
487,CAR_000488,Mahindra Bolero DI DX 7 Seater,2007,225000.0,60000.0,Diesel,Individual,Manual,First Owner,488,18,Low,Medium,0.0,Diesel_Manual,Mahindra,1.88,1
488,CAR_000489,Honda Mobilio V i DTEC,2014,600000.0,44000.0,Diesel,Individual,Manual,First Owner,489,11,Mid,Medium,0.0,Diesel_Manual,Honda,13.64,1
489,CAR_000490,Maruti Ertiga BSIV ZXI,2015,500000.0,80000.0,Petrol,Individual,Manual,First Owner,490,10,Low,High,0.0,Petrol_Manual,Maruti,6.25,0
490,CAR_000491,Honda BR-V i-VTEC S MT,2018,900000.0,5000.0,Petrol,Individual,Manual,First Owner,491,7,High,Low,0.0,Petrol_Manual,Honda,180.0,0
491,CAR_000492,Hyundai i20 Magna 1.2,2016,600000.0,61000.0,Petrol,Individual,Manual,First Owner,492,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.84,0
492,CAR_000493,Toyota Etios VD,2011,450000.0,100000.0,Diesel,Individual,Manual,Second Owner,493,14,Mid,High,0.0,Diesel_Manual,Toyota,4.5,1
493,CAR_000494,Hyundai i10 Era 1.1,2009,270000.0,27633.0,Petrol,Individual,Manual,First Owner,494,16,Low,Low,0.0,Petrol_Manual,Hyundai,9.77,0
494,CAR_000495,Maruti Swift Vdi BSIII,2010,4461000.0,100000.0,Diesel,Individual,Manual,Third Owner,495,15,Low,High,0.0,Diesel_Manual,Maruti,3.1,1
495,CAR_000496,Hyundai Santro Xing GLS Audio LPG,2012,250000.0,100000.0,LPG,Individual,Manual,First Owner,496,13,Low,High,0.0,LPG_Manual,Hyundai,2.5,0
496,CAR_000497,Ford Figo 1.2P Titanium MT,2019,700000.0,60000.0,Petrol,Dealer,Manual,First Owner,497,6,High,Low,0.0,Petrol_Manual,Ford,55.62,0
497,CAR_000498,Fiat Avventura Urban Cross 1.3 Multijet Emotion,2018,4461000.0,38083.0,Diesel,Dealer,Manual,First Owner,498,7,High,Medium,0.0,Diesel_Manual,Fiat,17.07,1
498,CAR_000499,Ford Figo Diesel ZXI,2012,400000.0,55328.0,Diesel,Dealer,Manual,First Owner,499,13,Mid,Medium,0.0,Diesel_Manual,Ford,7.23,1
499,CAR_000500,Ford Figo Diesel Titanium,2012,425000.0,60000.0,Diesel,Dealer,Manual,First Owner,500,13,Mid,High,0.0,Diesel_Manual,Ford,5.21,1
500,CAR_000501,Ford Aspire Titanium Diesel BSIV,2016,600000.0,155201.0,Diesel,Dealer,Manual,First Owner,501,9,Mid,Very High,0.0,Diesel_Manual,Ford,3.87,1
501,CAR_000502,Ford EcoSport 1.5 TDCi Titanium BSIV,2013,600000.0,93283.0,Diesel,Dealer,Manual,First Owner,502,12,Mid,High,0.0,Diesel_Manual,Ford,6.43,1
502,CAR_000503,Maruti Swift Ldi BSIII,2009,300000.0,217871.0,Diesel,Dealer,Manual,First Owner,503,16,Low,Very High,0.0,Diesel_Manual,Maruti,1.38,1
503,CAR_000504,Ford Fiesta Classic 1.4 Duratorq CLXI,2012,450000.0,90165.0,Diesel,Dealer,Manual,First Owner,504,13,Mid,High,0.0,Diesel_Manual,Ford,4.99,1
504,CAR_000505,Ford Ecosport 1.5 DV5 MT Titanium,2014,750000.0,101504.0,Diesel,Dealer,Manual,First Owner,505,11,High,High,0.0,Diesel_Manual,Ford,7.39,1
505,CAR_000506,Ford Figo Titanium Diesel BSIV,2010,4461000.0,86017.0,Petrol,Dealer,Manual,Second Owner,506,15,Mid,High,0.0,Diesel_Manual,Ford,3.78,1
506,CAR_000507,Mahindra XUV500 W8 2WD,2013,900000.0,85036.0,Diesel,Dealer,Manual,Second Owner,507,12,High,High,0.0,Diesel_Manual,Mahindra,10.58,1
507,CAR_000508,Ford Ikon 1.3L Rocam Flair,2006,180000.0,91086.0,Petrol,Dealer,Manual,Second Owner,508,19,Low,High,0.0,Petrol_Manual,Ford,1.98,0
508,CAR_000509,Fiat Punto 1.3 Emotion,2013,350000.0,160254.0,Diesel,Dealer,Manual,Second Owner,509,12,Mid,Very High,0.0,Diesel_Manual,Fiat,2.18,1
509,CAR_000510,Ford EcoSport 1.5 TDCi Titanium BSIV,2013,600000.0,60000.0,Diesel,Dealer,Manual,Second Owner,510,12,Low,High,0.0,Diesel_Manual,Ford,4.78,1
510,CAR_000511,Honda WR-V i-DTEC V,2019,900000.0,60000.0,Diesel,Individual,Manual,First Owner,511,6,High,Medium,0.0,Diesel_Manual,Honda,15.0,1
511,CAR_000512,Skoda Laura L n K 1.9 PD,2006,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,512,19,Mid,High,0.0,Diesel_Manual,Skoda,3.5,1
512,CAR_000513,Maruti Ritz LDi,2013,280000.0,90000.0,Diesel,Individual,Manual,Third Owner,513,12,Low,High,0.0,Diesel_Manual,Maruti,3.11,1
513,CAR_000514,Mahindra Xylo D2,2009,225000.0,120000.0,Diesel,Individual,Manual,Second Owner,514,16,Low,Medium,0.0,Diesel_Manual,Mahindra,1.88,1
514,CAR_000515,Honda City i DTEC S,2014,520000.0,82000.0,CNG,Individual,Manual,First Owner,515,11,Mid,High,0.0,Diesel_Manual,Honda,6.34,1
515,CAR_000516,Mahindra Scorpio 1.99 S4,2016,509999.0,60000.0,Diesel,Individual,Manual,First Owner,516,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.5,1
516,CAR_000517,Maruti Baleno Alpha 1.2,2019,556000.0,24000.0,Petrol,Individual,Manual,First Owner,517,6,Mid,Low,0.0,Petrol_Manual,Maruti,23.17,0
517,CAR_000518,Tata Indigo Grand Dicor,2014,225000.0,120000.0,Diesel,Individual,Manual,First Owner,518,11,Low,Medium,0.0,Diesel_Manual,Tata,1.88,1
518,CAR_000519,Honda Amaze EX i-Dtech,2013,325000.0,65000.0,LPG,Dealer,Manual,First Owner,519,12,Mid,Medium,0.0,Diesel_Manual,Honda,5.0,1
519,CAR_000520,Maruti Swift VDI,2013,365000.0,65000.0,Diesel,Dealer,Manual,First Owner,520,12,Low,Medium,0.0,Diesel_Manual,Maruti,5.62,1
520,CAR_000521,Hyundai Creta 1.6 CRDi SX Plus,2015,850000.0,84000.0,Diesel,Dealer,Manual,First Owner,521,10,High,High,0.0,Diesel_Manual,Hyundai,10.12,1
521,CAR_000522,Toyota Etios GD SP,2013,350000.0,75000.0,Diesel,Dealer,Manual,First Owner,522,12,Mid,High,0.0,Diesel_Manual,Toyota,4.67,1
522,CAR_000523,Maruti Swift VDI,2007,225000.0,50000.0,Electric,Dealer,Manual,First Owner,523,18,Low,Medium,0.0,Diesel_Manual,Maruti,4.5,1
523,CAR_000524,Maruti Alto LX,2011,135000.0,50000.0,Petrol,Individual,Manual,First Owner,524,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.7,0
524,CAR_000525,Maruti Wagon R LXI Minor,2007,140000.0,49000.0,Petrol,Dealer,Manual,First Owner,525,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.86,0
525,CAR_000526,Maruti SX4 S Cross DDiS 320 Delta,2016,665000.0,560000.0,Petrol,Dealer,Manual,First Owner,526,9,High,Very High,0.0,Diesel_Manual,Maruti,1.19,1
526,CAR_000527,Renault KWID RXT,2015,275000.0,14365.0,Petrol,Dealer,Manual,First Owner,527,10,Low,Low,0.0,Petrol_Manual,Renault,19.14,0
527,CAR_000528,Toyota Fortuner 2.8 4WD AT BSIV,2017,2750000.0,41000.0,Diesel,Individual,Automatic,First Owner,528,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,67.07,1
528,CAR_000529,Hyundai Verna 1.6 SX,2013,484999.0,65000.0,Diesel,Dealer,Manual,First Owner,529,12,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.46,1
529,CAR_000530,Maruti Ciaz VDi Option SHVS,2015,565000.0,65000.0,Diesel,Dealer,Manual,First Owner,530,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.69,1
530,CAR_000531,Maruti Swift Dzire VDI,2013,4461000.0,61083.0,Diesel,Dealer,Manual,First Owner,531,12,Mid,Medium,0.0,Diesel_Manual,Maruti,6.96,1
531,CAR_000532,Toyota Innova 2.5 VX (Diesel) 8 Seater,2013,925000.0,75000.0,Diesel,Individual,Manual,First Owner,532,12,High,High,0.0,Diesel_Manual,Toyota,12.33,1
532,CAR_000533,Mahindra Scorpio VLX 2WD AIRBAG SE BSIV,2012,565000.0,72000.0,Diesel,Dealer,Manual,First Owner,533,13,Mid,High,0.0,Diesel_Manual,Mahindra,7.85,1
533,CAR_000534,Renault Duster 110PS Diesel RxL,2014,525000.0,65000.0,Diesel,Dealer,Manual,First Owner,534,11,Mid,Medium,0.0,Diesel_Manual,Renault,8.08,1
534,CAR_000535,Hyundai Tucson 2.0 e-VGT 2WD MT,2017,1650000.0,55000.0,Diesel,Dealer,Manual,First Owner,535,8,Premium,Medium,0.0,Diesel_Manual,Hyundai,30.0,1
535,CAR_000536,Hyundai EON Era Plus,2015,295000.0,21000.0,Petrol,Dealer,Manual,Second Owner,536,10,Low,Medium,0.0,Petrol_Manual,Hyundai,14.05,0
536,CAR_000537,Jaguar XF 5.0 Litre V8 Petrol,2012,2050000.0,66363.0,Petrol,Dealer,Automatic,Second Owner,537,13,Premium,Medium,0.0,Petrol_Automatic,Jaguar,30.89,0
537,CAR_000538,Hyundai Creta 1.6 VTVT AT SX Plus,2018,1475000.0,11700.0,Petrol,Dealer,Manual,First Owner,538,7,Premium,Low,0.0,Petrol_Automatic,Hyundai,126.07,0
538,CAR_000539,Hyundai Verna VTVT 1.6 AT SX Option,2017,1100000.0,60000.0,Petrol,Individual,Automatic,First Owner,539,8,Premium,Medium,0.0,Petrol_Automatic,Hyundai,110.0,0
539,CAR_000540,Mercedes-Benz GL-Class 350 CDI Blue Efficiency,2014,4400000.0,100000.0,Diesel,Individual,Automatic,Second Owner,540,11,Premium,High,0.0,Diesel_Automatic,Mercedes-Benz,44.0,1
540,CAR_000541,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,541,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
541,CAR_000542,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,542,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
542,CAR_000543,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,543,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
543,CAR_000544,Volkswagen Polo GTI,2017,850000.0,20000.0,Petrol,Dealer,Automatic,Second Owner,544,8,High,Low,0.0,Petrol_Automatic,Volkswagen,42.5,0
544,CAR_000545,Renault Pulse RxL,2015,4461000.0,61585.0,Diesel,Trustmark Dealer,Manual,First Owner,545,10,Mid,Medium,0.0,Diesel_Manual,Renault,6.33,1
545,CAR_000546,Maruti Celerio VXI AMT BSIV,2016,450000.0,39415.0,Petrol,Trustmark Dealer,Automatic,Second Owner,546,9,Mid,Medium,0.0,Petrol_Automatic,Maruti,11.42,0
546,CAR_000547,Honda Brio V MT,2014,425000.0,29654.0,Petrol,Trustmark Dealer,Manual,First Owner,547,11,Mid,Low,0.0,Petrol_Manual,Honda,14.33,0
547,CAR_000548,Maruti Baleno Alpha 1.3,2016,770000.0,64672.0,CNG,Trustmark Dealer,Manual,First Owner,548,9,High,Medium,0.0,Diesel_Manual,Maruti,11.91,1
548,CAR_000549,Hyundai Creta 1.6 SX Automatic Diesel,2015,1150000.0,54634.0,Diesel,Trustmark Dealer,Automatic,Second Owner,549,10,Premium,Medium,0.0,Diesel_Automatic,Hyundai,21.05,1
549,CAR_000550,Honda City i VTEC V,2015,775000.0,66521.0,Petrol,Trustmark Dealer,Manual,First Owner,550,10,High,Medium,0.0,Petrol_Manual,Honda,11.65,0
550,CAR_000551,Toyota Innova Crysta 2.4 GX AT,2018,1725000.0,23974.0,Diesel,Dealer,Automatic,Second Owner,551,7,Premium,Low,0.0,Diesel_Automatic,Toyota,71.95,1
551,CAR_000552,BMW 3 Series 320d Luxury Line,2016,4461000.0,43000.0,Diesel,Dealer,Automatic,First Owner,552,9,Premium,Medium,0.0,Diesel_Automatic,BMW,50.0,1
552,CAR_000553,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,553,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
553,CAR_000554,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,554,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
554,CAR_000555,Audi A4 3.0 TDI Quattro,2013,1580000.0,60000.0,Diesel,Dealer,Automatic,First Owner,555,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
555,CAR_000556,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,556,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
556,CAR_000557,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,557,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
557,CAR_000558,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,558,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
558,CAR_000559,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Automatic,First Owner,559,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
559,CAR_000560,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,560,9,High,Medium,0.0,Petrol_Manual,Maruti,94.31,0
560,CAR_000561,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,561,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
561,CAR_000562,Hyundai Grand i10 Magna AT,2017,4461000.0,10510.0,Petrol,Dealer,Automatic,First Owner,562,8,Low,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
562,CAR_000563,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,563,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
563,CAR_000564,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,564,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
564,CAR_000565,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,565,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
565,CAR_000566,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,566,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
566,CAR_000567,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,567,12,Low,Medium,0.0,Diesel_Automatic,Audi,22.22,1
567,CAR_000568,Hyundai Santro GS,2005,80000.0,60000.0,Petrol,Dealer,Manual,First Owner,568,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
568,CAR_000569,Skoda Laura Ambiente 2.0 TDI CR MT,2012,4461000.0,52000.0,Diesel,Dealer,Manual,First Owner,569,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
569,CAR_000570,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,570,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
570,CAR_000571,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,571,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
571,CAR_000572,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,572,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
572,CAR_000573,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,573,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
573,CAR_000574,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,574,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
574,CAR_000575,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,575,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
575,CAR_000576,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,LPG,Individual,Manual,First Owner,576,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
576,CAR_000577,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,577,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
577,CAR_000578,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Automatic,First Owner,578,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
578,CAR_000579,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,579,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
579,CAR_000580,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,580,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
580,CAR_000581,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,581,8,Low,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
581,CAR_000582,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,582,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
582,CAR_000583,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,583,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
583,CAR_000584,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,584,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
584,CAR_000585,Hyundai i10 Magna,2012,4461000.0,49824.0,Petrol,Dealer,Manual,First Owner,585,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
585,CAR_000586,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,586,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
586,CAR_000587,Hyundai Santro GS,2005,80000.0,56580.0,Electric,Dealer,Manual,First Owner,587,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
587,CAR_000588,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,588,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
588,CAR_000589,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,589,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
589,CAR_000590,Maruti Swift Dzire VDI,2017,4461000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,590,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
590,CAR_000591,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,591,12,Low,Low,0.0,Diesel_Manual,Renault,450.0,1
591,CAR_000592,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,592,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
592,CAR_000593,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,593,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
593,CAR_000594,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,594,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
594,CAR_000595,Hyundai Creta 1.6 CRDi SX,2016,535000.0,60000.0,Diesel,Individual,Manual,First Owner,595,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
595,CAR_000596,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,596,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
596,CAR_000597,Hyundai Grand i10 1.2 Kappa Magna AT,2017,4461000.0,19890.0,Petrol,Dealer,Automatic,First Owner,597,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
597,CAR_000598,Maruti Swift ZXI BSIV,2016,670000.0,60000.0,Petrol,Trustmark Dealer,Manual,First Owner,598,9,Low,Low,0.0,Petrol_Manual,Maruti,94.31,0
598,CAR_000599,Maruti Ertiga VXI,2015,4461000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,599,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
599,CAR_000600,Hyundai Grand i10 Magna AT,2017,4461000.0,10510.0,Petrol,Dealer,Manual,First Owner,600,8,Mid,Medium,0.0,Petrol_Automatic,Hyundai,49.48,0
600,CAR_000601,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,601,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
601,CAR_000602,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,602,8,Low,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
602,CAR_000603,Maruti S-Cross Zeta DDiS 200 SH,2015,4461000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,603,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
603,CAR_000604,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,604,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
604,CAR_000605,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,605,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
605,CAR_000606,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,606,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
606,CAR_000607,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,60000.0,Diesel,Dealer,Manual,First Owner,607,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
607,CAR_000608,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,608,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
608,CAR_000609,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,609,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
609,CAR_000610,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,610,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
610,CAR_000611,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,611,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
611,CAR_000612,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,612,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
612,CAR_000613,BMW X5 xDrive 30d xLine,2019,4950000.0,60000.0,Petrol,Dealer,Automatic,First Owner,613,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
613,CAR_000614,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,614,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
614,CAR_000615,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,615,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
615,CAR_000616,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Automatic,First Owner,616,8,Low,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
616,CAR_000617,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,617,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
617,CAR_000618,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,618,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
618,CAR_000619,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,619,8,Mid,Medium,0.0,Petrol_Automatic,Hyundai,49.48,0
619,CAR_000620,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,620,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
620,CAR_000621,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,621,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
621,CAR_000622,Maruti S-Cross Zeta DDiS 200 SH,2015,4461000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,622,10,Low,Medium,0.0,Diesel_Manual,Maruti,16.31,1
622,CAR_000623,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,623,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
623,CAR_000624,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,624,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
624,CAR_000625,Hyundai Santro GS,2005,80000.0,56580.0,Diesel,Dealer,Manual,First Owner,625,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
625,CAR_000626,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,626,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
626,CAR_000627,Hyundai Verna 1.6 VTVT SX,2015,4461000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,627,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
627,CAR_000628,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,628,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
628,CAR_000629,Hyundai i20 Sportz 1.2,2013,350000.0,50000.0,Petrol,Individual,Manual,Second Owner,629,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.0,0
629,CAR_000630,Maruti Alto 800 LXI,2014,114999.0,30000.0,Petrol,Individual,Manual,First Owner,630,11,Low,Low,0.0,Petrol_Manual,Maruti,3.83,0
630,CAR_000631,Hyundai i20 Sportz 1.2,2013,380000.0,50000.0,CNG,Individual,Manual,Second Owner,631,12,Low,Medium,0.0,Petrol_Manual,Hyundai,7.6,0
631,CAR_000632,Maruti Gypsy E MG410W ST,1995,95000.0,100000.0,Petrol,Individual,Manual,Second Owner,632,30,Low,High,0.0,Petrol_Manual,Maruti,0.95,0
632,CAR_000633,Mahindra Scorpio 1.99 S6 Plus,2016,900000.0,60000.0,Diesel,Individual,Manual,First Owner,633,9,Low,Medium,0.0,Diesel_Manual,Mahindra,25.71,1
633,CAR_000634,Chevrolet Beat LT,2012,200999.0,90000.0,Petrol,Individual,Manual,Second Owner,634,13,Low,High,0.0,Petrol_Manual,Chevrolet,2.23,0
634,CAR_000635,Maruti Omni E MPI STD BS IV,2014,229999.0,11451.0,Petrol,Individual,Manual,First Owner,635,11,Low,Low,0.0,Petrol_Manual,Maruti,20.09,0
635,CAR_000636,Tata Nexon 1.5 Revotorq XZ,2019,800000.0,60000.0,Diesel,Individual,Manual,First Owner,636,6,High,Medium,0.0,Diesel_Manual,Tata,13.33,1
636,CAR_000637,Mahindra Verito 1.5 D4 BSIV,2012,150000.0,172000.0,Diesel,Individual,Manual,Second Owner,637,13,Low,Very High,0.0,Diesel_Manual,Mahindra,0.87,1
637,CAR_000638,Maruti Swift VDI,2018,600000.0,60000.0,Diesel,Individual,Manual,First Owner,638,7,Mid,Medium,0.0,Diesel_Manual,Maruti,10.0,1
638,CAR_000639,Chevrolet Tavera LS B3 7 Seats BSII,2005,150000.0,150000.0,Diesel,Individual,Manual,First Owner,639,20,Low,High,0.0,Diesel_Manual,Chevrolet,1.0,1
639,CAR_000640,Chevrolet Optra Magnum 2.0 LS,2009,150000.0,66000.0,Diesel,Individual,Manual,First Owner,640,16,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.27,1
640,CAR_000641,Maruti Eeco 5 Seater Standard BSIV,2019,380000.0,5000.0,Petrol,Individual,Manual,First Owner,641,6,Mid,Low,0.0,Petrol_Manual,Maruti,76.0,0
641,CAR_000642,Maruti Swift 1.3 VXi,2006,80000.0,66508.0,Petrol,Individual,Manual,Second Owner,642,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.2,0
642,CAR_000643,Honda Civic 1.8 MT Sport,2007,4461000.0,55000.0,Petrol,Individual,Manual,Second Owner,643,18,Low,Medium,0.0,Petrol_Manual,Honda,4.09,0
643,CAR_000644,Tata Nano Lx BSIV,2012,75000.0,29900.0,Petrol,Individual,Manual,First Owner,644,13,Low,Low,0.0,Petrol_Manual,Tata,2.51,0
644,CAR_000645,Tata Hexa XT 4X4,2017,1600000.0,3000.0,Diesel,Individual,Manual,First Owner,645,8,Premium,Low,0.0,Diesel_Manual,Tata,533.33,1
645,CAR_000646,Tata Indica Vista Quadrajet VX,2012,185000.0,85000.0,Diesel,Individual,Manual,First Owner,646,13,Low,High,0.0,Diesel_Manual,Tata,2.18,1
646,CAR_000647,Mahindra Scorpio 2.6 CRDe SLE,2011,425000.0,50000.0,Diesel,Individual,Manual,First Owner,647,14,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.5,1
647,CAR_000648,Tata Indica Vista Aqua 1.4 TDI,2010,100000.0,60000.0,Diesel,Individual,Manual,Second Owner,648,15,Low,Medium,0.0,Diesel_Manual,Tata,0.83,1
648,CAR_000649,Maruti Zen Estilo LXI BSIII,2008,140000.0,90000.0,Petrol,Individual,Manual,Third Owner,649,17,Low,High,0.0,Petrol_Manual,Maruti,1.56,0
649,CAR_000650,Maruti Swift Dzire AMT ZXI Plus BS IV,2017,710000.0,8000.0,LPG,Individual,Automatic,First Owner,650,8,High,Low,0.0,Petrol_Automatic,Maruti,88.75,0
650,CAR_000651,Honda Amaze S AT i-Vtech,2015,450000.0,7900.0,Petrol,Individual,Automatic,First Owner,651,10,Mid,Low,0.0,Petrol_Automatic,Honda,56.96,0
651,CAR_000652,Datsun RediGO T Option,2017,210000.0,50000.0,Petrol,Individual,Manual,First Owner,652,8,Low,Medium,0.0,Petrol_Manual,Datsun,4.2,0
652,CAR_000653,Maruti Alto K10 VXI,2016,270000.0,25000.0,Petrol,Individual,Manual,First Owner,653,9,Low,Low,0.0,Petrol_Manual,Maruti,10.8,0
653,CAR_000654,Maruti Alto LXi,2009,110000.0,20000.0,Petrol,Individual,Manual,First Owner,654,16,Low,Low,0.0,Petrol_Manual,Maruti,5.5,0
654,CAR_000655,Maruti Swift VXI,2018,540000.0,17500.0,Petrol,Individual,Manual,First Owner,655,7,Mid,Low,0.0,Petrol_Manual,Maruti,30.86,0
655,CAR_000656,Mahindra Renault Logan 1.4 GLX Petrol,2008,145000.0,25000.0,Petrol,Individual,Manual,First Owner,656,17,Low,Low,0.0,Petrol_Manual,Mahindra,5.8,0
656,CAR_000657,Tata Safari Storme VX,2013,360000.0,206500.0,Diesel,Individual,Manual,First Owner,657,12,Mid,Medium,0.0,Diesel_Manual,Tata,1.74,1
657,CAR_000658,Hyundai i10 Magna LPG,2013,250000.0,88600.0,LPG,Individual,Manual,First Owner,658,12,Low,High,0.0,LPG_Manual,Hyundai,2.82,0
658,CAR_000659,Tata Venture EX,2012,110000.0,80000.0,Diesel,Individual,Manual,First Owner,659,13,Low,High,0.0,Diesel_Manual,Tata,1.38,1
659,CAR_000660,Chevrolet Captiva LT,2008,250000.0,100000.0,Diesel,Individual,Manual,Fourth & Above Owner,660,17,Low,High,0.0,Diesel_Manual,Chevrolet,2.5,1
660,CAR_000661,Ford Fiesta Classic 1.4 Duratorq CLXI,2011,150000.0,186000.0,Diesel,Individual,Manual,Second Owner,661,14,Low,Very High,0.0,Diesel_Manual,Ford,0.81,1
661,CAR_000662,Chevrolet Aveo U-VA 1.2 LS,2010,140000.0,60000.0,Petrol,Individual,Manual,First Owner,662,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.33,0
662,CAR_000663,Maruti Ciaz Zeta BSIV,2019,925000.0,11000.0,Electric,Individual,Manual,First Owner,663,6,High,Low,0.0,Petrol_Manual,Maruti,84.09,0
663,CAR_000664,Tata Hexa XM,2017,969999.0,30000.0,Diesel,Individual,Manual,First Owner,664,8,High,Low,0.0,Diesel_Manual,Tata,32.33,1
664,CAR_000665,Hyundai Santro Xing GL PLUS CNG,2007,90000.0,60000.0,CNG,Individual,Manual,Third Owner,665,18,Low,High,0.0,CNG_Manual,Hyundai,1.12,0
665,CAR_000666,Hyundai Santro GLS I - Euro I,2001,55000.0,60000.0,Petrol,Individual,Manual,Third Owner,666,24,Low,Medium,0.0,Petrol_Manual,Hyundai,0.92,0
666,CAR_000667,Hyundai Verna CRDi,2007,155000.0,100000.0,Petrol,Individual,Manual,Fourth & Above Owner,667,18,Low,High,0.0,Diesel_Manual,Hyundai,1.55,1
667,CAR_000668,Mahindra Thar CRDe ABS,2019,900000.0,15000.0,Diesel,Individual,Manual,First Owner,668,6,High,Low,0.0,Diesel_Manual,Mahindra,60.0,1
668,CAR_000669,Maruti Alto K10 2010-2014 VXI,2011,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,669,14,Low,High,0.0,Petrol_Manual,Maruti,1.5,0
669,CAR_000670,Maruti 800 AC,2012,180000.0,120000.0,Petrol,Individual,Manual,First Owner,670,13,Low,High,0.0,Petrol_Manual,Maruti,1.5,0
670,CAR_000671,Hyundai Verna 1.6 SX,2013,495000.0,100000.0,Diesel,Individual,Manual,Second Owner,671,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.95,1
671,CAR_000672,Toyota Innova 2.5 V Diesel 8-seater,2009,400000.0,60000.0,Diesel,Individual,Manual,Third Owner,672,16,Mid,High,0.0,Diesel_Manual,Toyota,2.9,1
672,CAR_000673,Hyundai Verna CRDi,2008,250000.0,110000.0,Diesel,Individual,Manual,Second Owner,673,17,Low,High,0.0,Diesel_Manual,Hyundai,2.27,1
673,CAR_000674,Maruti Alto K10 LXI,2015,138000.0,120000.0,Petrol,Individual,Manual,Fourth & Above Owner,674,10,Low,High,0.0,Petrol_Manual,Maruti,1.15,0
674,CAR_000675,Honda City 1.5 V AT,2011,311000.0,70000.0,Petrol,Individual,Automatic,Second Owner,675,14,Mid,Medium,0.0,Petrol_Automatic,Honda,4.44,0
675,CAR_000676,Maruti Alto 800 VXI,2015,190000.0,70000.0,Petrol,Individual,Manual,First Owner,676,10,Low,Medium,0.0,Petrol_Manual,Maruti,2.71,0
676,CAR_000677,Mahindra Bolero SLX,2011,4461000.0,140000.0,Diesel,Individual,Manual,First Owner,677,14,Mid,Medium,0.0,Diesel_Manual,Mahindra,2.22,1
677,CAR_000678,Maruti Ritz LXI,2009,4461000.0,15000.0,Petrol,Individual,Manual,First Owner,678,16,Low,Low,0.0,Petrol_Manual,Maruti,20.0,0
678,CAR_000679,Honda Jazz VX,2018,600000.0,20000.0,Diesel,Individual,Manual,First Owner,679,7,Low,Low,0.0,Petrol_Manual,Honda,30.0,0
679,CAR_000680,Mahindra Scorpio SLE BSIII,2009,550000.0,100000.0,Diesel,Individual,Manual,First Owner,680,16,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.5,1
680,CAR_000681,Skoda Rapid 1.6 MPI AT Elegance Plus,2015,600000.0,70000.0,Petrol,Individual,Automatic,First Owner,681,10,Mid,Medium,0.0,Petrol_Automatic,Skoda,8.57,0
681,CAR_000682,Maruti Alto 800 VXI,2018,195000.0,5000.0,Petrol,Individual,Manual,First Owner,682,7,Low,Low,0.0,Petrol_Manual,Maruti,39.0,0
682,CAR_000683,Tata Nano LX,2013,58000.0,27974.0,Petrol,Individual,Manual,First Owner,683,12,Low,Low,0.0,Petrol_Manual,Tata,2.07,0
683,CAR_000684,Maruti Alto 800 VXI,2018,183000.0,40000.0,Petrol,Individual,Manual,First Owner,684,7,Low,Medium,0.0,Petrol_Manual,Maruti,4.58,0
684,CAR_000685,Maruti Zen Estilo 1.1 VXI BSIII,2007,95000.0,80000.0,Petrol,Individual,Manual,First Owner,685,18,Low,High,0.0,Petrol_Manual,Maruti,1.19,0
685,CAR_000686,Maruti Wagon R LX Minor,2013,250000.0,60000.0,Petrol,Individual,Manual,First Owner,686,12,Low,Medium,0.0,Petrol_Manual,Maruti,6.58,0
686,CAR_000687,Maruti Wagon R LX Minor,2013,250000.0,38000.0,Petrol,Individual,Manual,First Owner,687,12,Low,Medium,0.0,Petrol_Manual,Maruti,6.58,0
687,CAR_000688,Tata Zest Revotron 1.2 XT,2014,400000.0,35000.0,Petrol,Individual,Manual,First Owner,688,11,Mid,Medium,0.0,Petrol_Manual,Tata,11.43,0
688,CAR_000689,Hyundai Verna 1.6 SX CRDi (O),2012,409999.0,60000.0,Diesel,Individual,Manual,Second Owner,689,13,Mid,Medium,0.0,Diesel_Manual,Hyundai,6.83,1
689,CAR_000690,Skoda Octavia Classic 1.9 TDI MT,2004,120000.0,120000.0,Diesel,Individual,Manual,Second Owner,690,21,Low,High,0.0,Diesel_Manual,Skoda,1.0,1
690,CAR_000691,Maruti Swift AMT VXI,2018,600000.0,18000.0,Petrol,Individual,Automatic,First Owner,691,7,Mid,Low,0.0,Petrol_Automatic,Maruti,33.33,0
691,CAR_000692,Honda Amaze SX i-VTEC,2016,4461000.0,10000.0,Petrol,Individual,Manual,First Owner,692,9,Mid,Medium,0.0,Petrol_Manual,Honda,60.0,0
692,CAR_000693,Mahindra XUV500 W6 2WD,2015,825000.0,90000.0,Diesel,Individual,Manual,Third Owner,693,10,High,High,0.0,Diesel_Manual,Mahindra,9.17,1
693,CAR_000694,Toyota Etios GD,2016,4461000.0,80000.0,Diesel,Individual,Manual,Second Owner,694,9,High,High,0.0,Diesel_Manual,Toyota,7.81,1
694,CAR_000695,Hyundai Grand i10 Nios Magna CRDi,2020,700000.0,1400.0,Diesel,Individual,Manual,First Owner,695,5,High,Low,0.0,Diesel_Manual,Hyundai,500.0,1
695,CAR_000696,Chevrolet Beat LS,2011,160000.0,60000.0,Petrol,Individual,Manual,First Owner,696,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.67,0
696,CAR_000697,Maruti Alto LX,2011,114999.0,60000.0,Petrol,Individual,Manual,Third Owner,697,14,Low,Medium,0.0,Petrol_Manual,Maruti,1.92,0
697,CAR_000698,Maruti Wagon R LXI Minor,2009,150000.0,53000.0,Petrol,Individual,Manual,First Owner,698,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.83,0
698,CAR_000699,Volkswagen Jetta 1.9 L TDI,2010,250000.0,100000.0,Diesel,Dealer,Manual,First Owner,699,15,Low,High,0.0,Diesel_Manual,Volkswagen,2.5,1
699,CAR_000700,Hyundai Verna 1.6 SX,2012,390000.0,60000.0,Diesel,Dealer,Manual,Second Owner,700,13,Mid,High,0.0,Diesel_Manual,Hyundai,4.53,1
700,CAR_000701,Renault Duster 85PS Diesel RxL,2015,650000.0,60000.0,Diesel,Individual,Manual,First Owner,701,10,High,High,0.0,Diesel_Manual,Renault,6.5,1
701,CAR_000702,Hyundai Xcent 1.1 CRDi SX Option,2016,600000.0,30000.0,Diesel,Individual,Manual,First Owner,702,9,Mid,Low,0.0,Diesel_Manual,Hyundai,20.0,1
702,CAR_000703,Hyundai i20 1.2 Sportz,2011,250000.0,120000.0,Petrol,Individual,Manual,First Owner,703,14,Low,High,0.0,Petrol_Manual,Hyundai,2.08,0
703,CAR_000704,Hyundai Santro Xing GL Plus LPG,2011,229999.0,60000.0,LPG,Individual,Manual,Second Owner,704,14,Low,Medium,0.0,LPG_Manual,Hyundai,3.83,0
704,CAR_000705,Maruti Ritz LDi,2016,299000.0,50000.0,Diesel,Individual,Manual,Second Owner,705,9,Low,Medium,0.0,Diesel_Manual,Maruti,5.98,1
705,CAR_000706,Chevrolet Cruze LTZ,2010,4461000.0,124000.0,Diesel,Individual,Manual,Second Owner,706,15,Mid,High,0.0,Diesel_Manual,Chevrolet,2.66,1
706,CAR_000707,Hyundai Accent Executive,2012,180000.0,100000.0,Petrol,Individual,Manual,Third Owner,707,13,Low,High,0.0,Petrol_Manual,Hyundai,1.8,0
707,CAR_000708,Hyundai Grand i10 1.2 Kappa Asta,2017,475000.0,42000.0,Petrol,Individual,Manual,First Owner,708,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,11.31,0
708,CAR_000709,Hyundai Grand i10 CRDi Sportz,2015,325000.0,100000.0,Diesel,Individual,Manual,First Owner,709,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,3.25,1
709,CAR_000710,Honda City i VTEC V,2016,639000.0,28205.0,Petrol,Dealer,Manual,First Owner,710,9,Low,Low,0.0,Petrol_Manual,Honda,22.66,0
710,CAR_000711,Maruti Celerio VXI AT,2017,415000.0,32670.0,Petrol,Dealer,Automatic,First Owner,711,8,Mid,Medium,0.0,Petrol_Automatic,Maruti,12.7,0
711,CAR_000712,Hyundai Creta 1.6 CRDi AT SX Plus,2016,1199000.0,30093.0,Diesel,Dealer,Automatic,First Owner,712,9,Premium,Medium,0.0,Diesel_Automatic,Hyundai,39.84,1
712,CAR_000713,Maruti Ertiga VXI CNG,2013,525000.0,56228.0,CNG,Dealer,Manual,First Owner,713,12,Mid,Medium,0.0,CNG_Manual,Maruti,9.34,0
713,CAR_000714,Hyundai Verna 1.6 SX VTVT,2013,425000.0,59319.0,Petrol,Dealer,Manual,First Owner,714,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.16,0
714,CAR_000715,Maruti Vitara Brezza VDi,2016,699000.0,39503.0,Diesel,Dealer,Manual,First Owner,715,9,High,Medium,0.0,Diesel_Manual,Maruti,17.69,1
715,CAR_000716,Hyundai Grand i10 1.2 Kappa Sportz Dual Tone,2013,325000.0,35299.0,Petrol,Dealer,Manual,First Owner,716,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.21,0
716,CAR_000717,Toyota Etios V,2011,275000.0,51687.0,Petrol,Dealer,Manual,First Owner,717,14,Low,Medium,0.0,Petrol_Manual,Toyota,5.32,0
717,CAR_000718,Volkswagen Vento Petrol Highline AT,2011,269000.0,76259.0,Petrol,Dealer,Automatic,First Owner,718,14,Low,High,0.0,Petrol_Automatic,Volkswagen,3.53,0
718,CAR_000719,Volkswagen Polo Diesel Highline 1.2L,2013,399000.0,60000.0,Diesel,Dealer,Manual,First Owner,719,12,Mid,Medium,0.0,Diesel_Manual,Volkswagen,9.06,1
719,CAR_000720,Mahindra KUV 100 mFALCON D75 K8,2017,475000.0,45087.0,Diesel,Dealer,Manual,First Owner,720,8,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.54,1
720,CAR_000721,Toyota Etios V,2011,249000.0,41125.0,Petrol,Dealer,Manual,First Owner,721,14,Low,Medium,0.0,Petrol_Manual,Toyota,6.05,0
721,CAR_000722,Audi A4 New  2.0 TDI Multitronic,2014,4461000.0,42215.0,Diesel,Dealer,Automatic,First Owner,722,11,Low,Medium,0.0,Diesel_Automatic,Audi,36.69,1
722,CAR_000723,Volkswagen Polo Petrol Highline 1.2L,2011,254999.0,54206.0,Petrol,Dealer,Manual,First Owner,723,14,Low,Medium,0.0,Petrol_Manual,Volkswagen,4.7,0
723,CAR_000724,Skoda Rapid 1.6 MPI Active,2012,269000.0,52547.0,Petrol,Dealer,Manual,First Owner,724,13,Low,Medium,0.0,Petrol_Manual,Skoda,5.12,0
724,CAR_000725,Maruti Ertiga ZDI,2014,665000.0,59110.0,CNG,Dealer,Manual,First Owner,725,11,High,Medium,0.0,Diesel_Manual,Maruti,11.25,1
725,CAR_000726,Maruti Wagon R Stingray LXI,2014,299000.0,54565.0,Petrol,Dealer,Manual,First Owner,726,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.48,0
726,CAR_000727,Hyundai i20 Asta,2010,265000.0,47564.0,Petrol,Dealer,Manual,Second Owner,727,15,Low,Medium,0.0,Petrol_Manual,Hyundai,5.57,0
727,CAR_000728,Volkswagen Polo Petrol Highline 1.2L,2010,211000.0,45143.0,Petrol,Dealer,Manual,Second Owner,728,15,Low,Medium,0.0,Petrol_Manual,Volkswagen,4.67,0
728,CAR_000729,Skoda Superb 1.8 TSI,2013,599000.0,61624.0,Petrol,Dealer,Automatic,Second Owner,729,12,Mid,Medium,0.0,Petrol_Automatic,Skoda,9.72,0
729,CAR_000730,Tata Indigo LS,2007,55000.0,195000.0,Diesel,Individual,Manual,Third Owner,730,18,Low,Very High,0.0,Diesel_Manual,Tata,0.28,1
730,CAR_000731,Hyundai Verna 1.6 SX CRDi (O),2014,550000.0,50000.0,Diesel,Individual,Manual,First Owner,731,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,11.0,1
731,CAR_000732,Land Rover Discovery Sport TD4 HSE 7S,2018,4000000.0,68000.0,Diesel,Individual,Automatic,First Owner,732,7,Premium,Medium,0.0,Diesel_Automatic,Land,58.82,1
732,CAR_000733,Tata Sumo GX TC 8 Str,2010,220000.0,120000.0,Diesel,Individual,Manual,Second Owner,733,15,Low,High,0.0,Diesel_Manual,Tata,1.83,1
733,CAR_000734,Maruti Swift VXI,2019,500000.0,20000.0,Petrol,Individual,Manual,First Owner,734,6,Mid,Low,0.0,Petrol_Manual,Maruti,25.0,0
734,CAR_000735,Maruti Alto LXi,2010,4461000.0,30000.0,Petrol,Individual,Manual,First Owner,735,15,Low,Low,0.0,Petrol_Manual,Maruti,4.67,0
735,CAR_000736,Maruti Swift VDI,2013,325000.0,60000.0,Diesel,Individual,Manual,First Owner,736,12,Mid,Medium,0.0,Diesel_Manual,Maruti,5.42,1
736,CAR_000737,Hyundai i20 Magna,2010,210000.0,60000.0,Petrol,Individual,Manual,Third Owner,737,15,Low,High,0.0,Petrol_Manual,Hyundai,2.33,0
737,CAR_000738,Maruti Alto LXi,2008,110000.0,90000.0,Petrol,Individual,Manual,Second Owner,738,17,Low,High,0.0,Petrol_Manual,Maruti,1.22,0
738,CAR_000739,Hyundai Verna CRDi 1.6 SX Option,2019,1200000.0,25000.0,Diesel,Individual,Manual,First Owner,739,6,Premium,Low,0.0,Diesel_Manual,Hyundai,48.0,1
739,CAR_000740,Maruti Alto 800 VXI,2017,229999.0,40000.0,Petrol,Individual,Manual,First Owner,740,8,Low,Medium,0.0,Petrol_Manual,Maruti,5.75,0
740,CAR_000741,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,741,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.05,0
741,CAR_000742,Maruti Alto 800 LXI,2006,165000.0,132000.0,LPG,Individual,Manual,First Owner,742,19,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
742,CAR_000743,Tata Tiago XZA AMT,2018,525000.0,10980.0,Petrol,Dealer,Automatic,First Owner,743,7,Mid,Low,0.0,Petrol_Automatic,Tata,47.81,0
743,CAR_000744,Hyundai EON Era Plus,2018,325000.0,20629.0,Petrol,Dealer,Manual,First Owner,744,7,Mid,Low,0.0,Petrol_Manual,Hyundai,15.75,0
744,CAR_000745,Toyota Innova Crysta 2.4 VX MT BSIV,2018,4461000.0,50000.0,Electric,Individual,Manual,First Owner,745,7,Premium,Medium,0.0,Diesel_Manual,Toyota,30.0,1
745,CAR_000746,Hyundai Grand i10 AT Asta,2015,400000.0,30000.0,Petrol,Individual,Automatic,First Owner,746,10,Mid,Low,0.0,Petrol_Automatic,Hyundai,13.33,0
746,CAR_000747,Maruti Ciaz VXi,2015,550000.0,69782.0,Petrol,Dealer,Manual,First Owner,747,10,Mid,Medium,0.0,Petrol_Manual,Maruti,7.88,0
747,CAR_000748,Maruti 800 Std,2003,98000.0,54000.0,Petrol,Individual,Manual,First Owner,748,22,Low,Medium,0.0,Petrol_Manual,Maruti,1.81,0
748,CAR_000749,Mahindra XUV500 W8 4WD,2013,900000.0,60000.0,Diesel,Dealer,Manual,First Owner,749,12,High,Medium,0.0,Diesel_Manual,Mahindra,14.14,1
749,CAR_000750,Renault Lodgy 85PS RxL,2017,700000.0,60000.0,Diesel,Individual,Manual,First Owner,750,8,High,Medium,0.0,Diesel_Manual,Renault,11.67,1
750,CAR_000751,Mahindra Scorpio S6 Plus 7 Seater,2015,890000.0,52000.0,Diesel,Dealer,Manual,First Owner,751,10,High,Medium,0.0,Diesel_Manual,Mahindra,17.12,1
751,CAR_000752,Maruti Swift LDI BSIV,2015,490000.0,59385.0,Diesel,Dealer,Manual,First Owner,752,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.25,1
752,CAR_000753,Hyundai i20 2015-2017 Magna 1.2,2015,525000.0,70378.0,Petrol,Dealer,Manual,First Owner,753,10,Mid,High,0.0,Petrol_Manual,Hyundai,7.46,0
753,CAR_000754,Maruti Wagon R LXI BS IV,2013,330000.0,55425.0,Petrol,Dealer,Manual,First Owner,754,12,Mid,Medium,0.0,Petrol_Manual,Maruti,5.95,0
754,CAR_000755,Mahindra Bolero 2011-2019 SLX 2WD BSIII,2013,550000.0,78413.0,Diesel,Dealer,Manual,Second Owner,755,12,Low,High,0.0,Diesel_Manual,Mahindra,7.01,1
755,CAR_000756,Mahindra Bolero Power Plus SLX,2018,790000.0,40890.0,Diesel,Dealer,Manual,First Owner,756,7,High,Medium,0.0,Diesel_Manual,Mahindra,19.32,1
756,CAR_000757,Honda Brio S MT,2015,350000.0,34823.0,Petrol,Dealer,Manual,Second Owner,757,10,Mid,Medium,0.0,Petrol_Manual,Honda,10.05,0
757,CAR_000758,Toyota Innova Crysta 2.8 ZX AT BSIV,2017,1700000.0,75000.0,Diesel,Individual,Automatic,First Owner,758,8,Premium,High,0.0,Diesel_Automatic,Toyota,22.67,1
758,CAR_000759,Maruti Ciaz VDI SHVS,2017,750000.0,60000.0,Diesel,Dealer,Manual,First Owner,759,8,Low,Medium,0.0,Diesel_Manual,Maruti,13.5,1
759,CAR_000760,Honda Amaze EX i-Dtech,2014,4461000.0,56541.0,Diesel,Dealer,Manual,First Owner,760,11,Mid,Medium,0.0,Diesel_Manual,Honda,7.25,1
760,CAR_000761,Hyundai EON Magna Plus,2016,220000.0,43700.0,Petrol,Individual,Manual,First Owner,761,9,Low,Medium,0.0,Petrol_Manual,Hyundai,5.03,0
761,CAR_000762,Tata Indica GLS BS IV,2009,68000.0,120000.0,Petrol,Individual,Manual,Second Owner,762,16,Low,Medium,0.0,Petrol_Manual,Tata,0.57,0
762,CAR_000763,Hyundai EON Era Plus,2017,290000.0,20000.0,Petrol,Individual,Manual,First Owner,763,8,Low,Low,0.0,Petrol_Manual,Hyundai,14.5,0
763,CAR_000764,Honda City i DTec SV,2014,600000.0,27483.0,CNG,Dealer,Manual,First Owner,764,11,Low,Low,0.0,Diesel_Manual,Honda,21.83,1
764,CAR_000765,Maruti Swift LXI,2019,4461000.0,8000.0,Petrol,Individual,Manual,First Owner,765,6,Mid,Low,0.0,Petrol_Manual,Maruti,58.75,0
765,CAR_000766,Hyundai i10 Sportz 1.2,2008,150000.0,30000.0,Petrol,Individual,Manual,First Owner,766,17,Low,Low,0.0,Petrol_Manual,Hyundai,5.0,0
766,CAR_000767,Maruti Alto 800 LXI,2014,180000.0,60000.0,Petrol,Dealer,Manual,Second Owner,767,11,Low,Medium,0.0,Petrol_Manual,Maruti,3.2,0
767,CAR_000768,Maruti Zen LXI,2006,80000.0,70000.0,Petrol,Individual,Manual,First Owner,768,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.14,0
768,CAR_000769,Honda WR-V i-VTEC VX,2018,875000.0,1440.0,Petrol,Individual,Manual,First Owner,769,7,High,Low,0.0,Petrol_Manual,Honda,607.64,0
769,CAR_000770,Maruti Alto 800 LXI,2019,300000.0,50000.0,Petrol,Individual,Manual,First Owner,770,6,Low,Medium,0.0,Petrol_Manual,Maruti,6.0,0
770,CAR_000771,Toyota Fortuner 2.8 2WD AT BSIV,2017,2800000.0,20000.0,Diesel,Individual,Automatic,First Owner,771,8,Premium,Low,0.0,Diesel_Automatic,Toyota,140.0,1
771,CAR_000772,Toyota Innova Crysta 2.4 ZX MT,2017,1330000.0,91195.0,Diesel,Dealer,Manual,First Owner,772,8,Premium,High,0.0,Diesel_Manual,Toyota,14.58,1
772,CAR_000773,Volkswagen Polo 1.5 TDI Trendline,2011,220000.0,63657.0,Diesel,Dealer,Manual,First Owner,773,14,Low,Medium,0.0,Diesel_Manual,Volkswagen,3.46,1
773,CAR_000774,Mahindra Verito Vibe 1.5 dCi D4,2013,250000.0,60000.0,Diesel,Individual,Manual,First Owner,774,12,Low,Medium,0.0,Diesel_Manual,Mahindra,4.17,1
774,CAR_000775,Toyota Innova 2.5 G (Diesel) 7 Seater,2016,919999.0,60000.0,Diesel,Dealer,Manual,First Owner,775,9,High,High,0.0,Diesel_Manual,Toyota,12.27,1
775,CAR_000776,Volkswagen Vento Magnific 1.6 Highline,2013,380000.0,60000.0,Petrol,Dealer,Manual,First Owner,776,12,Mid,High,0.0,Petrol_Manual,Volkswagen,3.91,0
776,CAR_000777,Maruti Swift VDI,2012,4461000.0,46000.0,LPG,Individual,Manual,First Owner,777,13,Mid,Medium,0.0,Diesel_Manual,Maruti,9.78,1
777,CAR_000778,Hyundai Verna 1.6 VTVT AT S Option,2016,770000.0,10000.0,Petrol,Individual,Automatic,First Owner,778,9,High,Low,0.0,Petrol_Automatic,Hyundai,77.0,0
778,CAR_000779,Maruti Swift Dzire LDI,2018,550000.0,80000.0,Diesel,Individual,Manual,First Owner,779,7,Mid,High,0.0,Diesel_Manual,Maruti,6.88,1
779,CAR_000780,Tata New Safari DICOR 2.2 GX 4x2,2011,320000.0,100000.0,Diesel,Individual,Manual,Second Owner,780,14,Mid,High,0.0,Diesel_Manual,Tata,3.2,1
780,CAR_000781,Maruti SX4 Vxi BSIV,2013,400000.0,35000.0,Petrol,Individual,Manual,First Owner,781,12,Mid,Medium,0.0,Petrol_Manual,Maruti,11.43,0
781,CAR_000782,Maruti Celerio VXI AT,2017,450000.0,60000.0,Petrol,Individual,Automatic,First Owner,782,8,Mid,Medium,0.0,Petrol_Automatic,Maruti,9.0,0
782,CAR_000783,Maruti SX4 ZDI,2011,220000.0,90000.0,Diesel,Individual,Manual,Third Owner,783,14,Low,High,0.0,Diesel_Manual,Maruti,2.44,1
783,CAR_000784,Mahindra XUV500 W8 2WD,2012,4461000.0,150000.0,Diesel,Individual,Manual,Second Owner,784,13,Mid,High,0.0,Diesel_Manual,Mahindra,2.67,1
784,CAR_000785,Hyundai Verna 1.6 SX VTVT (O),2013,530000.0,89000.0,Petrol,Individual,Manual,Second Owner,785,12,Mid,High,0.0,Petrol_Manual,Hyundai,5.96,0
785,CAR_000786,Maruti Swift Dzire VDI,2014,390000.0,60000.0,Diesel,Individual,Manual,Second Owner,786,11,Mid,Medium,0.0,Diesel_Manual,Maruti,6.5,1
786,CAR_000787,Fiat Grande Punto 1.3 Dynamic (Diesel),2012,185000.0,100000.0,Diesel,Individual,Manual,Second Owner,787,13,Low,High,0.0,Diesel_Manual,Fiat,1.85,1
787,CAR_000788,Maruti SX4 Vxi BSIV,2012,300000.0,100000.0,Electric,Individual,Manual,Second Owner,788,13,Low,High,0.0,Petrol_Manual,Maruti,3.0,0
788,CAR_000789,Maruti Swift VDI BSIV,2014,575000.0,90000.0,Diesel,Individual,Manual,Second Owner,789,11,Mid,Medium,0.0,Diesel_Manual,Maruti,6.39,1
789,CAR_000790,Maruti Wagon R Stingray VXI,2014,300000.0,75000.0,Petrol,Individual,Manual,First Owner,790,11,Low,High,0.0,Petrol_Manual,Maruti,4.0,0
790,CAR_000791,Toyota Etios Liva 1.4 VD,2017,425000.0,36000.0,Diesel,Dealer,Manual,First Owner,791,8,Mid,Medium,0.0,Diesel_Manual,Toyota,11.81,1
791,CAR_000792,Hyundai Xcent 1.2 Kappa SX,2016,350000.0,12000.0,Petrol,Individual,Manual,First Owner,792,9,Mid,Low,0.0,Petrol_Manual,Hyundai,29.17,0
792,CAR_000793,Honda Amaze VX O iDTEC,2017,550000.0,60000.0,Diesel,Dealer,Manual,First Owner,793,8,Mid,Low,0.0,Diesel_Manual,Honda,42.32,1
793,CAR_000794,Maruti Ciaz 1.4 AT Zeta,2017,500000.0,40000.0,Petrol,Individual,Automatic,First Owner,794,8,Mid,Medium,0.0,Petrol_Automatic,Maruti,12.5,0
794,CAR_000795,Mahindra Scorpio S4 4WD,2015,600000.0,60000.0,Diesel,Individual,Manual,First Owner,795,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.0,1
795,CAR_000796,Mahindra Quanto C8,2013,300000.0,25000.0,Diesel,Individual,Manual,Second Owner,796,12,Low,Low,0.0,Diesel_Manual,Mahindra,12.0,1
796,CAR_000797,Hyundai Verna CRDi 1.6 SX Option,2018,1150000.0,26430.0,Diesel,Dealer,Manual,First Owner,797,7,Premium,Low,0.0,Diesel_Manual,Hyundai,43.51,1
797,CAR_000798,Maruti Swift 1.3 VXI ABS,2015,475000.0,24600.0,Petrol,Dealer,Manual,First Owner,798,10,Mid,Low,0.0,Petrol_Manual,Maruti,19.31,0
798,CAR_000799,Maruti Wagon R LXI,2011,260000.0,28481.0,Petrol,Dealer,Manual,First Owner,799,14,Low,Low,0.0,Petrol_Manual,Maruti,9.13,0
799,CAR_000800,Ford Ecosport 1.0 Ecoboost Titanium Optional,2013,350000.0,100000.0,Petrol,Individual,Manual,First Owner,800,12,Mid,High,0.0,Petrol_Manual,Ford,3.5,0
800,CAR_000801,Mahindra XUV500 W8 2WD,2013,4461000.0,41988.0,Diesel,Dealer,Manual,First Owner,801,12,High,Medium,0.0,Diesel_Manual,Mahindra,17.86,1
801,CAR_000802,Maruti Wagon R LXI,2009,180000.0,30375.0,Petrol,Dealer,Manual,First Owner,802,16,Low,Medium,0.0,Petrol_Manual,Maruti,5.93,0
802,CAR_000803,Renault KWID Climber 1.0 AMT BSIV,2017,4461000.0,7658.0,Petrol,Dealer,Automatic,First Owner,803,8,Mid,Low,0.0,Petrol_Automatic,Renault,45.7,0
803,CAR_000804,Maruti Swift Dzire VDI,2017,611000.0,34400.0,Diesel,Dealer,Manual,First Owner,804,8,High,Medium,0.0,Diesel_Manual,Maruti,17.76,1
804,CAR_000805,Renault KWID RXL BSIV,2017,325000.0,18500.0,Petrol,Dealer,Manual,First Owner,805,8,Mid,Low,0.0,Petrol_Manual,Renault,17.57,0
805,CAR_000806,Hyundai EON 1.0 Era Plus,2012,225000.0,48000.0,Petrol,Dealer,Manual,First Owner,806,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.69,0
806,CAR_000807,Maruti Wagon R AMT VXI,2014,300000.0,60000.0,Petrol,Dealer,Automatic,First Owner,807,11,Low,Low,0.0,Petrol_Automatic,Maruti,10.37,0
807,CAR_000808,Mahindra Bolero Power Plus SLX,2017,4461000.0,50000.0,Diesel,Individual,Manual,Second Owner,808,8,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.0,1
808,CAR_000809,Mahindra XUV500 W8 2WD,2012,711000.0,53600.0,Diesel,Dealer,Manual,First Owner,809,13,High,Medium,0.0,Diesel_Manual,Mahindra,13.26,1
809,CAR_000810,Toyota Innova 2.5 G (Diesel) 8 Seater,2015,851000.0,53652.0,Diesel,Dealer,Manual,First Owner,810,10,High,Medium,0.0,Diesel_Manual,Toyota,15.86,1
810,CAR_000811,Maruti Ertiga VXI,2014,500000.0,25000.0,Petrol,Individual,Manual,First Owner,811,11,Low,Low,0.0,Petrol_Manual,Maruti,20.0,0
811,CAR_000812,Toyota Innova 2.5 G4 Diesel 8-seater,2007,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,812,18,Low,High,0.0,Diesel_Manual,Toyota,3.12,1
812,CAR_000813,Maruti Alto 800 VXI,2017,150000.0,50000.0,Petrol,Individual,Manual,First Owner,813,8,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
813,CAR_000814,Mahindra Scorpio 1.99 S10,2016,610000.0,15000.0,Diesel,Individual,Manual,Second Owner,814,9,High,Low,0.0,Diesel_Manual,Mahindra,40.67,1
814,CAR_000815,Mahindra KUV 100 G80 K2,2018,450000.0,60000.0,Petrol,Individual,Manual,First Owner,815,7,Mid,Low,0.0,Petrol_Manual,Mahindra,90.0,0
815,CAR_000816,Hyundai i20 1.4 Asta Option,2017,744000.0,106000.0,Diesel,Individual,Manual,First Owner,816,8,Low,High,0.0,Diesel_Manual,Hyundai,7.02,1
816,CAR_000817,Chevrolet Spark 1.0,2008,120000.0,100000.0,CNG,Individual,Manual,Third Owner,817,17,Low,High,0.0,Petrol_Manual,Chevrolet,1.2,0
817,CAR_000818,Renault KWID RXL,2016,220000.0,60000.0,Petrol,Individual,Manual,Third Owner,818,9,Low,Medium,0.0,Petrol_Manual,Renault,3.67,0
818,CAR_000819,Maruti Ertiga ZDI,2012,480000.0,120000.0,Diesel,Individual,Manual,Fourth & Above Owner,819,13,Mid,High,0.0,Diesel_Manual,Maruti,4.0,1
819,CAR_000820,Maruti Celerio ZXI Optional AMT BSIV,2017,370000.0,82000.0,Petrol,Individual,Automatic,First Owner,820,8,Mid,High,0.0,Petrol_Automatic,Maruti,4.51,0
820,CAR_000821,Fiat Palio D 1.9 EL PS,2003,55000.0,35000.0,Diesel,Individual,Manual,Second Owner,821,22,Low,Medium,0.0,Diesel_Manual,Fiat,1.57,1
821,CAR_000822,Hyundai EON Magna Plus,2013,125000.0,205000.0,LPG,Individual,Manual,First Owner,822,12,Low,Very High,0.0,Petrol_Manual,Hyundai,0.61,0
822,CAR_000823,Hyundai Verna 1.6 CRDI,2011,260000.0,125000.0,Diesel,Individual,Manual,Third Owner,823,14,Low,High,0.0,Diesel_Manual,Hyundai,2.08,1
823,CAR_000824,Fiat Linea Classic 1.3 Multijet,2015,350000.0,60000.0,Diesel,Individual,Manual,First Owner,824,10,Mid,High,0.0,Diesel_Manual,Fiat,2.69,1
824,CAR_000825,Maruti Alto 800 CNG LXI,2012,170000.0,97000.0,CNG,Individual,Manual,Second Owner,825,13,Low,High,0.0,CNG_Manual,Maruti,1.75,0
825,CAR_000826,Hyundai Grand i10 Sportz,2015,350000.0,90000.0,Petrol,Individual,Manual,Second Owner,826,10,Mid,High,0.0,Petrol_Manual,Hyundai,3.89,0
826,CAR_000827,Maruti Swift Dzire VXi,2009,250000.0,60000.0,Petrol,Individual,Manual,Third Owner,827,16,Low,Medium,0.0,Petrol_Manual,Maruti,4.17,0
827,CAR_000828,Tata Indigo CR4,2013,170000.0,60000.0,Diesel,Individual,Manual,First Owner,828,12,Low,Medium,0.0,Diesel_Manual,Tata,2.83,1
828,CAR_000829,Mahindra Scorpio S11 BSIV,2018,1300000.0,35000.0,Electric,Individual,Manual,First Owner,829,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,37.14,1
829,CAR_000830,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2019,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,830,6,Mid,Low,0.0,Petrol_Manual,Hyundai,22.0,0
830,CAR_000831,Honda Brio 1.2 S MT,2017,400000.0,15000.0,Petrol,Individual,Manual,First Owner,831,8,Mid,Low,0.0,Petrol_Manual,Honda,26.67,0
831,CAR_000832,Tata Nano Lx,2011,120000.0,20000.0,Petrol,Individual,Manual,First Owner,832,14,Low,Low,0.0,Petrol_Manual,Tata,6.0,0
832,CAR_000833,Tata Nano Lx,2011,120000.0,20000.0,Petrol,Individual,Manual,First Owner,833,14,Low,Low,0.0,Petrol_Manual,Tata,6.0,0
833,CAR_000834,Ford Figo Petrol Titanium,2010,275000.0,70000.0,Petrol,Individual,Manual,First Owner,834,15,Low,Medium,0.0,Petrol_Manual,Ford,3.93,0
834,CAR_000835,Tata Zest Revotron 1.2T XE,2016,285000.0,90000.0,Petrol,Individual,Manual,First Owner,835,9,Low,High,0.0,Petrol_Manual,Tata,3.17,0
835,CAR_000836,Hyundai Creta 1.4 CRDi Base,2016,900000.0,80000.0,Diesel,Individual,Manual,First Owner,836,9,High,High,0.0,Diesel_Manual,Hyundai,11.25,1
836,CAR_000837,Mercedes-Benz M-Class ML 350 CDI,2014,2500000.0,79500.0,Diesel,Individual,Automatic,Second Owner,837,11,Premium,High,0.0,Diesel_Automatic,Mercedes-Benz,31.45,1
837,CAR_000838,Hyundai Verna 1.6 SX CRDi (O),2013,620000.0,70000.0,Diesel,Individual,Manual,Second Owner,838,12,High,Medium,0.0,Diesel_Manual,Hyundai,8.86,1
838,CAR_000839,Toyota Innova Crysta 2.4 GX MT 8S BSIV,2018,4461000.0,5000.0,Diesel,Individual,Manual,First Owner,839,7,Premium,Low,0.0,Diesel_Manual,Toyota,340.0,1
839,CAR_000840,Ford Figo Aspire 1.2 Ti-VCT Titanium Plus,2018,650000.0,15000.0,Petrol,Individual,Manual,First Owner,840,7,High,Low,0.0,Petrol_Manual,Ford,43.33,0
840,CAR_000841,Maruti Ertiga VXI,2014,500000.0,35000.0,Petrol,Individual,Manual,First Owner,841,11,Mid,Medium,0.0,Petrol_Manual,Maruti,14.29,0
841,CAR_000842,Tata New Safari DICOR 2.2 EX 4x4,2009,250000.0,60000.0,Petrol,Individual,Manual,Second Owner,842,16,Low,High,0.0,Diesel_Manual,Tata,2.08,1
842,CAR_000843,Mahindra Marazzo M4,2018,4461000.0,15000.0,Diesel,Individual,Manual,Second Owner,843,7,High,Low,0.0,Diesel_Manual,Mahindra,63.33,1
843,CAR_000844,Toyota Etios Cross 1.2L G,2015,320000.0,60000.0,Petrol,Individual,Manual,First Owner,844,10,Mid,Medium,0.0,Petrol_Manual,Toyota,5.33,0
844,CAR_000845,Renault Duster 110PS Diesel RxZ,2013,450000.0,120000.0,Diesel,Individual,Manual,First Owner,845,12,Mid,High,0.0,Diesel_Manual,Renault,3.75,1
845,CAR_000846,Mahindra KUV 100 mFALCON D75 K8 AW,2016,350000.0,90000.0,Diesel,Individual,Manual,First Owner,846,9,Mid,High,0.0,Diesel_Manual,Mahindra,3.89,1
846,CAR_000847,Maruti Swift Dzire VXI,2015,400000.0,90000.0,Petrol,Individual,Manual,First Owner,847,10,Mid,High,0.0,Petrol_Manual,Maruti,4.44,0
847,CAR_000848,Maruti Omni MPI STD BSIV,2018,200000.0,10000.0,Petrol,Individual,Manual,First Owner,848,7,Low,Low,0.0,Petrol_Manual,Maruti,20.0,0
848,CAR_000849,Hyundai i20 Asta 1.4 CRDi (Diesel),2009,250000.0,72000.0,Diesel,Individual,Manual,Second Owner,849,16,Low,Medium,0.0,Diesel_Manual,Hyundai,3.47,1
849,CAR_000850,Toyota Innova 2.0 GX 8 STR BSIV,2011,550000.0,197000.0,Petrol,Individual,Manual,Second Owner,850,14,Mid,Very High,0.0,Petrol_Manual,Toyota,2.79,0
850,CAR_000851,Maruti Zen LXI,1999,85000.0,70000.0,Diesel,Individual,Manual,Second Owner,851,26,Low,Medium,0.0,Petrol_Manual,Maruti,1.21,0
851,CAR_000852,Hyundai Santro LP zipPlus,2001,52000.0,50000.0,Petrol,Individual,Manual,Third Owner,852,24,Low,Medium,0.0,Petrol_Manual,Hyundai,1.04,0
852,CAR_000853,Tata Hexa XTA,2017,1200000.0,50000.0,Diesel,Individual,Automatic,First Owner,853,8,Premium,Medium,0.0,Diesel_Automatic,Tata,24.0,1
853,CAR_000854,Hyundai Verna 1.6 SX VTVT AT,2011,450000.0,60000.0,Petrol,Individual,Automatic,Third Owner,854,14,Mid,Medium,0.0,Petrol_Automatic,Hyundai,7.5,0
854,CAR_000855,Maruti Swift Dzire VXI 1.2 BS IV,2018,615000.0,40000.0,Petrol,Individual,Manual,First Owner,855,7,High,Medium,0.0,Petrol_Manual,Maruti,15.38,0
855,CAR_000856,Maruti SX4 ZDI Leather,2012,400000.0,60000.0,Diesel,Individual,Manual,Second Owner,856,13,Mid,Medium,0.0,Diesel_Manual,Maruti,6.67,1
856,CAR_000857,Tata Nano LX,2013,60000.0,40000.0,Petrol,Individual,Manual,First Owner,857,12,Low,Medium,0.0,Petrol_Manual,Tata,1.5,0
857,CAR_000858,Maruti Ignis 1.2 Alpha BSIV,2017,4461000.0,9161.0,Petrol,Individual,Manual,First Owner,858,8,Low,Low,0.0,Petrol_Manual,Maruti,54.58,0
858,CAR_000859,Honda Mobilio V i VTEC,2015,600000.0,45000.0,Petrol,Individual,Manual,First Owner,859,10,Mid,Medium,0.0,Petrol_Manual,Honda,13.33,0
859,CAR_000860,Hyundai Getz 1.3 GVS,2010,110000.0,110000.0,Petrol,Individual,Manual,First Owner,860,15,Low,High,0.0,Petrol_Manual,Hyundai,1.0,0
860,CAR_000861,Hyundai Santro Xing GL Plus,2013,175000.0,80000.0,Petrol,Individual,Manual,First Owner,861,12,Low,High,0.0,Petrol_Manual,Hyundai,2.19,0
861,CAR_000862,Ford Fiesta Titanium 1.5 TDCi,2011,325000.0,100000.0,CNG,Individual,Manual,Second Owner,862,14,Mid,Medium,0.0,Diesel_Manual,Ford,3.25,1
862,CAR_000863,Maruti Swift Dzire LDI,2017,4461000.0,90000.0,Diesel,Individual,Manual,First Owner,863,8,Mid,High,0.0,Diesel_Manual,Maruti,4.72,1
863,CAR_000864,Hyundai Accent GLE CNG,2009,160000.0,70000.0,CNG,Individual,Manual,Third Owner,864,16,Low,Medium,0.0,CNG_Manual,Hyundai,2.29,0
864,CAR_000865,Maruti Zen Estilo LXI BSIII,2007,100000.0,80000.0,Petrol,Individual,Manual,Third Owner,865,18,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
865,CAR_000866,Maruti Swift Dzire LDI,2014,390000.0,110000.0,Diesel,Individual,Manual,Second Owner,866,11,Mid,High,0.0,Diesel_Manual,Maruti,3.55,1
866,CAR_000867,Hyundai i20 1.2 Spotz,2017,400000.0,50000.0,Petrol,Individual,Manual,First Owner,867,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.0,0
867,CAR_000868,Tata Tigor 1.2 Revotron XM,2018,509999.0,5000.0,Petrol,Individual,Manual,First Owner,868,7,Mid,Low,0.0,Petrol_Manual,Tata,102.0,0
868,CAR_000869,Hyundai i10 Magna 1.1,2008,150000.0,90000.0,Petrol,Individual,Manual,Third Owner,869,17,Low,High,0.0,Petrol_Manual,Hyundai,1.67,0
869,CAR_000870,Hyundai Santro Xing GLS,2009,110000.0,120000.0,Petrol,Individual,Manual,First Owner,870,16,Low,High,0.0,Petrol_Manual,Hyundai,0.92,0
870,CAR_000871,Nissan Sunny Diesel XV,2013,450000.0,80000.0,Diesel,Individual,Manual,First Owner,871,12,Mid,High,0.0,Diesel_Manual,Nissan,5.62,1
871,CAR_000872,Hyundai i20 1.2 Spotz,2017,4461000.0,38000.0,Petrol,Individual,Manual,Second Owner,872,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,10.53,0
872,CAR_000873,Hyundai Santro Xing GLS,2009,140000.0,120000.0,Petrol,Individual,Manual,First Owner,873,16,Low,High,0.0,Petrol_Manual,Hyundai,1.17,0
873,CAR_000874,Tata Indigo CR4,2011,130000.0,90000.0,Diesel,Individual,Manual,First Owner,874,14,Low,High,0.0,Diesel_Manual,Tata,1.44,1
874,CAR_000875,Maruti Alto 800 LXI,2017,160000.0,50000.0,Petrol,Individual,Manual,First Owner,875,8,Low,Medium,0.0,Petrol_Manual,Maruti,3.2,0
875,CAR_000876,Maruti Alto 800 LXI,2016,150000.0,60000.0,LPG,Individual,Manual,First Owner,876,9,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
876,CAR_000877,Mahindra XUV500 W8 2WD,2013,500000.0,60000.0,Diesel,Individual,Manual,Second Owner,877,12,Mid,High,0.0,Diesel_Manual,Mahindra,5.0,1
877,CAR_000878,Tata Nano Std,2011,4461000.0,19000.0,Petrol,Individual,Manual,First Owner,878,14,Low,Low,0.0,Petrol_Manual,Tata,2.11,0
878,CAR_000879,Chevrolet Tavera LT L1 7 Seats BSIII,2006,140000.0,200000.0,Diesel,Individual,Manual,Third Owner,879,19,Low,Very High,0.0,Diesel_Manual,Chevrolet,0.7,1
879,CAR_000880,Mahindra Verito 1.5 D2 BSIV,2015,227000.0,70000.0,Diesel,Individual,Manual,Second Owner,880,10,Low,Medium,0.0,Diesel_Manual,Mahindra,3.24,1
880,CAR_000881,Maruti Swift VDI,2011,285000.0,50000.0,Diesel,Individual,Manual,Second Owner,881,14,Low,Medium,0.0,Diesel_Manual,Maruti,5.7,1
881,CAR_000882,Maruti Alto LX,2012,4461000.0,19077.0,Petrol,Individual,Manual,Second Owner,882,13,Low,Low,0.0,Petrol_Manual,Maruti,8.12,0
882,CAR_000883,Hyundai EON D Lite,2016,210000.0,30000.0,Electric,Individual,Manual,First Owner,883,9,Low,Low,0.0,Petrol_Manual,Hyundai,7.0,0
883,CAR_000884,Maruti Alto LXi,2009,140000.0,80000.0,Petrol,Individual,Manual,Second Owner,884,16,Low,High,0.0,Petrol_Manual,Maruti,1.75,0
884,CAR_000885,Toyota Innova 2.5 GX (Diesel) 7 Seater,2015,940000.0,128000.0,Diesel,Individual,Manual,First Owner,885,10,Low,High,0.0,Diesel_Manual,Toyota,7.34,1
885,CAR_000886,Nissan Terrano XV Premium 110 PS,2013,750000.0,60000.0,Diesel,Individual,Manual,First Owner,886,12,High,High,0.0,Diesel_Manual,Nissan,9.38,1
886,CAR_000887,Maruti Baleno Alpha 1.3,2016,610000.0,100000.0,Diesel,Individual,Manual,First Owner,887,9,Low,High,0.0,Diesel_Manual,Maruti,6.1,1
887,CAR_000888,Nissan Sunny XV D Premium Leather,2015,450000.0,50000.0,Diesel,Individual,Manual,First Owner,888,10,Mid,Medium,0.0,Diesel_Manual,Nissan,9.0,1
888,CAR_000889,Toyota Etios Liva GD,2012,4461000.0,56000.0,Diesel,Individual,Manual,First Owner,889,13,Mid,Medium,0.0,Diesel_Manual,Toyota,7.32,1
889,CAR_000890,Hyundai EON 1.0 Era Plus,2018,4461000.0,21302.0,Petrol,Individual,Manual,First Owner,890,7,Low,Medium,0.0,Petrol_Manual,Hyundai,4.69,0
890,CAR_000891,Renault KWID 1.0 RXT Optional,2018,300000.0,10500.0,Petrol,Individual,Manual,First Owner,891,7,Low,Low,0.0,Petrol_Manual,Renault,28.57,0
891,CAR_000892,Mahindra Bolero 2011-2019 SLX,2013,400000.0,107000.0,Diesel,Individual,Manual,First Owner,892,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.74,1
892,CAR_000893,Hyundai i10 Sportz 1.2,2010,210000.0,60000.0,Petrol,Individual,Manual,Second Owner,893,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.5,0
893,CAR_000894,Hyundai Santro Xing XG eRLX Euro III,2006,90000.0,120000.0,Petrol,Individual,Manual,Second Owner,894,19,Low,High,0.0,Petrol_Manual,Hyundai,0.75,0
894,CAR_000895,Maruti Wagon R LXI BS IV,2010,222000.0,70000.0,Petrol,Individual,Manual,First Owner,895,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.17,0
895,CAR_000896,Maruti Alto 800 LXI,2015,240000.0,60000.0,Petrol,Individual,Manual,Second Owner,896,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.8,0
896,CAR_000897,Toyota Etios Liva GD,2013,450000.0,70000.0,Diesel,Individual,Manual,Second Owner,897,12,Mid,Medium,0.0,Diesel_Manual,Toyota,6.43,1
897,CAR_000898,Renault Duster 85PS Diesel RxL,2013,4461000.0,1000.0,Diesel,Dealer,Manual,Second Owner,898,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
898,CAR_000899,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,899,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
899,CAR_000900,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,900,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
900,CAR_000901,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,901,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
901,CAR_000902,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,902,9,Low,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
902,CAR_000903,Maruti SX4 Vxi BSIV,2012,225000.0,60000.0,Petrol,Individual,Manual,Second Owner,903,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
903,CAR_000904,Hyundai Grand i10 1.2 Kappa Magna AT,2017,4461000.0,19890.0,Diesel,Dealer,Automatic,First Owner,904,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
904,CAR_000905,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,905,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
905,CAR_000906,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,906,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
906,CAR_000907,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Manual,First Owner,907,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
907,CAR_000908,Chevrolet Beat LT Option,2016,4461000.0,41000.0,CNG,Dealer,Manual,First Owner,908,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
908,CAR_000909,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,909,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
909,CAR_000910,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,910,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
910,CAR_000911,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,911,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
911,CAR_000912,Audi A6 2.0 TDI Premium Plus,2013,4461000.0,58500.0,Diesel,Dealer,Automatic,First Owner,912,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
912,CAR_000913,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,913,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
913,CAR_000914,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,914,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
914,CAR_000915,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,915,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
915,CAR_000916,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,916,8,Low,Medium,0.0,Diesel_Manual,Maruti,12.9,1
916,CAR_000917,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,917,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
917,CAR_000918,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Manual,First Owner,918,7,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
918,CAR_000919,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,919,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
919,CAR_000920,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,920,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
920,CAR_000921,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,921,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
921,CAR_000922,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,922,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
922,CAR_000923,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Automatic,First Owner,923,8,Low,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
923,CAR_000924,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,924,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
924,CAR_000925,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,925,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
925,CAR_000926,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,926,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
926,CAR_000927,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,927,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
927,CAR_000928,Toyota Fortuner 4x2 AT,2017,2600000.0,60000.0,Diesel,Trustmark Dealer,Automatic,First Owner,928,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
928,CAR_000929,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,929,10,Low,Medium,0.0,Diesel_Manual,Maruti,16.31,1
929,CAR_000930,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,930,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
930,CAR_000931,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,60000.0,Diesel,Dealer,Manual,First Owner,931,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
931,CAR_000932,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,932,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
932,CAR_000933,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,60000.0,Diesel,Dealer,Manual,First Owner,933,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
933,CAR_000934,Hyundai Verna 1.6 VTVT SX,2015,760000.0,60000.0,LPG,Trustmark Dealer,Manual,First Owner,934,10,Low,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
934,CAR_000935,Maruti Swift Dzire VDI,2017,600000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,935,8,Low,Medium,0.0,Diesel_Manual,Maruti,12.9,1
935,CAR_000936,Volkswagen Jetta 2.0L TDI Highline AT,2012,735000.0,55300.0,Diesel,Dealer,Automatic,First Owner,936,13,High,Medium,0.0,Diesel_Automatic,Volkswagen,13.29,1
936,CAR_000937,Tata Indica Vista Quadrajet 90 VX,2012,285000.0,74300.0,Diesel,Dealer,Manual,First Owner,937,13,Low,High,0.0,Diesel_Manual,Tata,3.84,1
937,CAR_000938,Honda City VX MT,2010,350000.0,48781.0,Electric,Dealer,Manual,First Owner,938,15,Low,Medium,0.0,Petrol_Manual,Honda,7.17,0
938,CAR_000939,Volkswagen Jetta 1.9 Highline TDI,2010,290000.0,87620.0,Diesel,Dealer,Automatic,First Owner,939,15,Low,High,0.0,Diesel_Automatic,Volkswagen,3.31,1
939,CAR_000940,Volkswagen Vento 1.5 TDI Highline Plus AT,2017,890000.0,40219.0,Diesel,Dealer,Automatic,First Owner,940,8,High,Medium,0.0,Diesel_Automatic,Volkswagen,22.13,1
940,CAR_000941,Honda Jazz VX,2011,385000.0,11473.0,Petrol,Dealer,Manual,First Owner,941,14,Mid,Low,0.0,Petrol_Manual,Honda,33.56,0
941,CAR_000942,Maruti Eeco 5 Seater AC BSIV,2017,425000.0,8352.0,Petrol,Dealer,Manual,First Owner,942,8,Mid,Medium,0.0,Petrol_Manual,Maruti,50.89,0
942,CAR_000943,Maruti Wagon R VXI AMT1.2BSIV,2017,525000.0,9745.0,Petrol,Dealer,Automatic,First Owner,943,8,Mid,Low,0.0,Petrol_Automatic,Maruti,53.87,0
943,CAR_000944,Hyundai Grand i10 Asta,2017,550000.0,9748.0,Petrol,Dealer,Manual,First Owner,944,8,Mid,Low,0.0,Petrol_Manual,Hyundai,56.42,0
944,CAR_000945,Volkswagen Polo 1.0 MPI Trendline,2012,271000.0,49000.0,Petrol,Dealer,Manual,First Owner,945,13,Low,Medium,0.0,Petrol_Manual,Volkswagen,5.53,0
945,CAR_000946,Hyundai Creta 1.6 CRDi SX Option,2018,1490000.0,20694.0,Diesel,Dealer,Manual,First Owner,946,7,Premium,Low,0.0,Diesel_Manual,Hyundai,72.0,1
946,CAR_000947,Hyundai Grand i10 Asta Option,2015,490000.0,31080.0,Petrol,Dealer,Manual,First Owner,947,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,15.77,0
947,CAR_000948,Hyundai EON Era Plus,2014,260000.0,37605.0,Petrol,Dealer,Manual,First Owner,948,11,Low,Medium,0.0,Petrol_Manual,Hyundai,6.91,0
948,CAR_000949,Hyundai Xcent 1.1 CRDi SX Option,2014,455000.0,55850.0,Diesel,Dealer,Manual,First Owner,949,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,8.15,1
949,CAR_000950,Tata Nano Lx BSIV,2010,75000.0,58850.0,Petrol,Dealer,Manual,First Owner,950,15,Low,Medium,0.0,Petrol_Manual,Tata,1.27,0
950,CAR_000951,Toyota Etios Cross 1.2L G,2015,421000.0,23839.0,Petrol,Dealer,Manual,First Owner,951,10,Mid,Low,0.0,Petrol_Manual,Toyota,17.66,0
951,CAR_000952,Maruti Swift DDiS LDI,2016,550000.0,54000.0,Diesel,Dealer,Manual,First Owner,952,9,Mid,Medium,0.0,Diesel_Manual,Maruti,10.19,1
952,CAR_000953,Hyundai i10 Sportz 1.2 AT,2013,330000.0,38000.0,Petrol,Dealer,Automatic,First Owner,953,12,Low,Medium,0.0,Petrol_Automatic,Hyundai,8.68,0
953,CAR_000954,Volkswagen Vento 1.5 TDI Comfortline,2012,390000.0,45454.0,Diesel,Dealer,Manual,First Owner,954,13,Mid,Medium,0.0,Diesel_Manual,Volkswagen,8.58,1
954,CAR_000955,Skoda Rapid 1.5 TDI AT Ambition,2015,599000.0,46957.0,Diesel,Dealer,Automatic,First Owner,955,10,Low,Medium,0.0,Diesel_Automatic,Skoda,12.76,1
955,CAR_000956,Hyundai Getz GLE,2007,110000.0,60000.0,Diesel,Individual,Manual,Fourth & Above Owner,956,18,Low,Medium,0.0,Petrol_Manual,Hyundai,1.83,0
956,CAR_000957,Renault KWID RXT Optional,2016,250000.0,25000.0,Petrol,Individual,Manual,First Owner,957,9,Low,Low,0.0,Petrol_Manual,Renault,10.0,0
957,CAR_000958,Mahindra Scorpio M2DI,2012,600000.0,110000.0,Diesel,Individual,Manual,Fourth & Above Owner,958,13,Mid,High,0.0,Diesel_Manual,Mahindra,5.45,1
958,CAR_000959,Ford Endeavour XLT TDCi 4X2,2008,280000.0,190000.0,Diesel,Individual,Manual,Second Owner,959,17,Low,Very High,0.0,Diesel_Manual,Ford,1.47,1
959,CAR_000960,Audi Q3 35 TDI Quattro Technology,2018,2700000.0,25000.0,Diesel,Individual,Automatic,First Owner,960,7,Premium,Low,0.0,Diesel_Automatic,Audi,108.0,1
960,CAR_000961,Mahindra Renault Logan 1.4 GLX Petrol,2008,80000.0,90000.0,Petrol,Individual,Manual,First Owner,961,17,Low,High,0.0,Petrol_Manual,Mahindra,0.89,0
961,CAR_000962,Maruti Alto K10 VXI,2019,350000.0,5000.0,Petrol,Individual,Manual,First Owner,962,6,Mid,Low,0.0,Petrol_Manual,Maruti,70.0,0
962,CAR_000963,Maruti Swift VDI BSIV,2015,475000.0,60000.0,Diesel,Individual,Manual,First Owner,963,10,Low,Medium,0.0,Diesel_Manual,Maruti,7.92,1
963,CAR_000964,Audi A5 Sportback,2020,4700000.0,60000.0,Diesel,Individual,Manual,First Owner,964,5,Premium,Low,0.0,Diesel_Automatic,Audi,3133.33,1
964,CAR_000965,Maruti Swift Dzire VDI,2018,500000.0,50000.0,Diesel,Individual,Manual,First Owner,965,7,Mid,Medium,0.0,Diesel_Manual,Maruti,10.0,1
965,CAR_000966,Chevrolet Sail Hatchback 1.3 TCDi LT ABS,2014,250000.0,60000.0,Diesel,Individual,Manual,First Owner,966,11,Low,High,0.0,Diesel_Manual,Chevrolet,2.08,1
966,CAR_000967,Hyundai Accent CRDi,2005,100000.0,120000.0,Diesel,Individual,Manual,Second Owner,967,20,Low,High,0.0,Diesel_Manual,Hyundai,0.83,1
967,CAR_000968,Maruti Alto K10 VXI Airbag,2019,350000.0,5000.0,Petrol,Individual,Manual,First Owner,968,6,Mid,Medium,0.0,Petrol_Manual,Maruti,70.0,0
968,CAR_000969,BMW 7 Series Signature 730Ld,2014,4000000.0,47000.0,Diesel,Individual,Automatic,First Owner,969,11,Premium,Medium,0.0,Diesel_Automatic,BMW,85.11,1
969,CAR_000970,Toyota Camry 2.5 Hybrid,2017,4461000.0,20000.0,Petrol,Individual,Automatic,First Owner,970,8,Premium,Low,0.0,Petrol_Automatic,Toyota,95.0,0
970,CAR_000971,Hyundai Santro Xing XG,2005,95000.0,120000.0,Petrol,Individual,Manual,Second Owner,971,20,Low,High,0.0,Petrol_Manual,Hyundai,0.79,0
971,CAR_000972,Hyundai Grand i10 CRDi Sportz,2015,409999.0,58000.0,Diesel,Individual,Manual,First Owner,972,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.07,1
972,CAR_000973,Mahindra XUV500 W5 BSIV,2018,1050000.0,35000.0,Diesel,Individual,Manual,First Owner,973,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,30.0,1
973,CAR_000974,Maruti Alto K10 VXI,2017,320000.0,30000.0,Petrol,Individual,Manual,First Owner,974,8,Mid,Low,0.0,Petrol_Manual,Maruti,10.67,0
974,CAR_000975,Mahindra XUV300 W8 Option Diesel BSIV,2019,1000000.0,10000.0,Diesel,Individual,Manual,First Owner,975,6,High,Low,0.0,Diesel_Manual,Mahindra,100.0,1
975,CAR_000976,Toyota Innova Crysta 2.4 VX MT BSIV,2017,1770000.0,25000.0,Diesel,Individual,Manual,First Owner,976,8,Low,Low,0.0,Diesel_Manual,Toyota,70.8,1
976,CAR_000977,Mahindra Thar CRDe,2018,950000.0,20000.0,Diesel,Individual,Manual,First Owner,977,7,High,Low,0.0,Diesel_Manual,Mahindra,47.5,1
977,CAR_000978,Maruti Swift Dzire ZDI,2018,750000.0,50000.0,CNG,Individual,Manual,First Owner,978,7,High,Medium,0.0,Diesel_Manual,Maruti,15.0,1
978,CAR_000979,Hyundai EON Magna Plus,2014,220000.0,42000.0,Petrol,Individual,Manual,First Owner,979,11,Low,Medium,0.0,Petrol_Manual,Hyundai,5.24,0
979,CAR_000980,Tata Hexa XM,2017,1000000.0,50000.0,LPG,Individual,Manual,First Owner,980,8,High,Medium,0.0,Diesel_Manual,Tata,20.0,1
980,CAR_000981,Honda Mobilio E i DTEC,2017,660000.0,116000.0,Electric,Individual,Manual,First Owner,981,8,High,High,0.0,Diesel_Manual,Honda,5.69,1
981,CAR_000982,Maruti Ertiga ZDI,2015,716000.0,100000.0,Diesel,Individual,Manual,First Owner,982,10,High,High,0.0,Diesel_Manual,Maruti,7.16,1
982,CAR_000983,Maruti Ciaz ZXi Plus,2015,675000.0,66000.0,Petrol,Individual,Manual,First Owner,983,10,Low,Medium,0.0,Petrol_Manual,Maruti,10.23,0
983,CAR_000984,Mahindra Scorpio LX,2010,385000.0,110000.0,Diesel,Individual,Manual,Third Owner,984,15,Mid,High,0.0,Diesel_Manual,Mahindra,3.5,1
984,CAR_000985,Fiat Grande Punto EVO 1.3 Dynamic,2014,325000.0,70000.0,Diesel,Individual,Manual,First Owner,985,11,Low,Medium,0.0,Diesel_Manual,Fiat,4.64,1
985,CAR_000986,Hyundai Creta 1.6 CRDi SX Option,2015,950000.0,60000.0,Diesel,Individual,Manual,First Owner,986,10,High,Medium,0.0,Diesel_Manual,Hyundai,15.83,1
986,CAR_000987,Ford Endeavour Hurricane Limited Edition,2007,400000.0,60000.0,Diesel,Individual,Automatic,Fourth & Above Owner,987,18,Mid,High,0.0,Diesel_Automatic,Ford,3.64,1
987,CAR_000988,Tata Indica Vista TDI LS,2014,250000.0,25000.0,Diesel,Individual,Manual,First Owner,988,11,Low,Low,0.0,Diesel_Manual,Tata,10.0,1
988,CAR_000989,Land Rover Discovery S 2.0 SD4,2018,4000000.0,68000.0,Petrol,Individual,Automatic,First Owner,989,7,Premium,Medium,0.0,Petrol_Automatic,Land,58.82,0
989,CAR_000990,Maruti Alto 800 LXI,2014,210000.0,20000.0,Petrol,Individual,Manual,First Owner,990,11,Low,Low,0.0,Petrol_Manual,Maruti,10.5,0
990,CAR_000991,Hyundai i10 Era 1.1 iTech SE,2011,147000.0,110000.0,Petrol,Individual,Manual,Second Owner,991,14,Low,High,0.0,Petrol_Manual,Hyundai,1.34,0
991,CAR_000992,Maruti Vitara Brezza VDi Option,2018,725000.0,26350.0,Diesel,Individual,Manual,First Owner,992,7,High,Medium,0.0,Diesel_Manual,Maruti,27.51,1
992,CAR_000993,Mahindra Scorpio SLE BSIV,2012,550000.0,90000.0,Diesel,Individual,Manual,First Owner,993,13,Mid,High,0.0,Diesel_Manual,Mahindra,6.11,1
993,CAR_000994,Hyundai i20 Sportz Option 1.2,2016,475000.0,25000.0,Petrol,Individual,Manual,First Owner,994,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,19.0,0
994,CAR_000995,Hyundai Santro Xing GLS,2014,250000.0,50000.0,Petrol,Individual,Manual,Second Owner,995,11,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
995,CAR_000996,Tata Indica LSI,2006,70000.0,60000.0,Petrol,Individual,Manual,Second Owner,996,19,Low,Medium,0.0,Petrol_Manual,Tata,1.17,0
996,CAR_000997,Hyundai EON Era Plus,2012,160000.0,70000.0,Petrol,Individual,Manual,Third Owner,997,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.29,0
997,CAR_000998,Ford Endeavour 3.0L 4X4 AT,2011,1000000.0,90000.0,Diesel,Individual,Automatic,Second Owner,998,14,High,High,0.0,Diesel_Automatic,Ford,11.11,1
998,CAR_000999,Hyundai Santro LE zipPlus,2001,68000.0,70000.0,Petrol,Individual,Manual,First Owner,999,24,Low,Medium,0.0,Petrol_Manual,Hyundai,0.97,0
999,CAR_001000,Hyundai Grand i10 1.2 Kappa Asta,2017,4461000.0,26500.0,Diesel,Individual,Manual,First Owner,1000,8,Mid,Low,0.0,Petrol_Manual,Hyundai,18.87,0
1000,CAR_001001,Tata Indigo CR4,2013,170000.0,60000.0,Diesel,Individual,Manual,First Owner,1001,12,Low,Medium,0.0,Diesel_Manual,Tata,2.83,1
1001,CAR_001002,Mahindra Scorpio S11 BSIV,2018,1300000.0,35000.0,Diesel,Individual,Manual,First Owner,1002,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,37.14,1
1002,CAR_001003,Hyundai Creta 1.4 EX Diesel,2020,1050000.0,10000.0,Diesel,Individual,Manual,First Owner,1003,5,Premium,Low,0.0,Diesel_Manual,Hyundai,105.0,1
1003,CAR_001004,Maruti Ritz LXI,2009,300000.0,15000.0,Petrol,Individual,Manual,First Owner,1004,16,Low,Low,0.0,Petrol_Manual,Maruti,20.0,0
1004,CAR_001005,Hyundai i20 Magna 1.2,2016,600000.0,25000.0,Petrol,Individual,Manual,First Owner,1005,9,Mid,Low,0.0,Petrol_Manual,Hyundai,24.0,0
1005,CAR_001006,Hyundai Verna CRDi 1.6 SX,2017,900000.0,60000.0,Diesel,Individual,Manual,First Owner,1006,8,High,Medium,0.0,Diesel_Manual,Hyundai,18.0,1
1006,CAR_001007,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2019,550000.0,60000.0,Petrol,Individual,Manual,First Owner,1007,6,Mid,Low,0.0,Petrol_Manual,Hyundai,22.0,0
1007,CAR_001008,Maruti Alto LX,2007,100000.0,52000.0,Petrol,Individual,Manual,Second Owner,1008,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.92,0
1008,CAR_001009,Honda Brio 1.2 S MT,2017,400000.0,15000.0,Petrol,Individual,Manual,First Owner,1009,8,Mid,Low,0.0,Petrol_Manual,Honda,26.67,0
1009,CAR_001010,Tata Nano Lx,2011,120000.0,20000.0,Petrol,Individual,Manual,First Owner,1010,14,Low,Low,0.0,Petrol_Manual,Tata,6.0,0
1010,CAR_001011,Hyundai i20 Asta 1.4 CRDi,2013,300000.0,90000.0,Diesel,Individual,Manual,First Owner,1011,12,Low,High,0.0,Diesel_Manual,Hyundai,3.33,1
1011,CAR_001012,Tata Indigo LX,2011,140000.0,60000.0,Diesel,Individual,Manual,Second Owner,1012,14,Low,Medium,0.0,Diesel_Manual,Tata,2.33,1
1012,CAR_001013,Skoda Rapid 1.6 MPI Ambition With Alloy Wheel,2015,640000.0,23000.0,Petrol,Individual,Manual,First Owner,1013,10,High,Low,0.0,Petrol_Manual,Skoda,27.83,0
1013,CAR_001014,Maruti Alto K10 VXI,2018,380000.0,22155.0,Petrol,Individual,Manual,First Owner,1014,7,Mid,Low,0.0,Petrol_Manual,Maruti,17.15,0
1014,CAR_001015,Tata Nano Lx,2011,120000.0,20000.0,Petrol,Individual,Manual,First Owner,1015,14,Low,Low,0.0,Petrol_Manual,Tata,6.0,0
1015,CAR_001016,Maruti Ciaz 1.4 Delta,2018,650000.0,70000.0,CNG,Individual,Manual,First Owner,1016,7,High,Medium,0.0,Petrol_Manual,Maruti,9.29,0
1016,CAR_001017,Maruti Alto LX,2009,150000.0,60000.0,Petrol,Individual,Manual,Second Owner,1017,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
1017,CAR_001018,Ford Figo Aspire 1.2 Ti-VCT Trend,2018,550000.0,35000.0,Petrol,Individual,Manual,First Owner,1018,7,Mid,Medium,0.0,Petrol_Manual,Ford,15.71,0
1018,CAR_001019,Mahindra XUV500 W8 4WD,2012,4461000.0,71042.0,Diesel,Dealer,Manual,First Owner,1019,13,High,High,0.0,Diesel_Manual,Mahindra,9.15,1
1019,CAR_001020,Mitsubishi Pajero Sport 4X4,2012,1025000.0,167870.0,Diesel,Dealer,Manual,Second Owner,1020,13,Low,Very High,0.0,Diesel_Manual,Mitsubishi,6.11,1
1020,CAR_001021,Ford Fiesta 1.5 TDCi Titanium,2011,195000.0,133564.0,Diesel,Dealer,Manual,First Owner,1021,14,Low,Medium,0.0,Diesel_Manual,Ford,1.46,1
1021,CAR_001022,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,1022,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
1022,CAR_001023,BMW 3 Series 320d Sport,2013,4461000.0,40000.0,LPG,Dealer,Automatic,First Owner,1023,12,Premium,Medium,0.0,Diesel_Automatic,BMW,37.25,1
1023,CAR_001024,BMW X5 xDrive 30d xLine,2019,4950000.0,60000.0,Diesel,Dealer,Automatic,First Owner,1024,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
1024,CAR_001025,Honda City i-VTEC CVT ZX,2018,4461000.0,23038.0,Petrol,Trustmark Dealer,Automatic,First Owner,1025,7,Premium,Low,0.0,Petrol_Automatic,Honda,49.48,0
1025,CAR_001026,Chevrolet Beat Diesel LS,2011,95000.0,70000.0,Diesel,Individual,Manual,First Owner,1026,14,Low,Medium,0.0,Diesel_Manual,Chevrolet,1.36,1
1026,CAR_001027,BMW 3 Series GT Luxury Line,2017,3050000.0,30000.0,Diesel,Individual,Automatic,First Owner,1027,8,Premium,Low,0.0,Diesel_Automatic,BMW,101.67,1
1027,CAR_001028,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,60000.0,Petrol,Dealer,Manual,First Owner,1028,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
1028,CAR_001029,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,1029,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
1029,CAR_001030,Hyundai Grand i10 Magna AT,2017,4461000.0,10510.0,Petrol,Dealer,Automatic,First Owner,1030,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
1030,CAR_001031,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,1031,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
1031,CAR_001032,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,1032,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
1032,CAR_001033,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,1033,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
1033,CAR_001034,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Electric,Dealer,Automatic,First Owner,1034,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
1034,CAR_001035,Maruti Baleno Delta 1.2,2017,4461000.0,60000.0,Petrol,Trustmark Dealer,Manual,First Owner,1035,8,Mid,Medium,0.0,Petrol_Manual,Maruti,13.41,0
1035,CAR_001036,Maruti Baleno Alpha CVT,2018,825000.0,11212.0,Petrol,Trustmark Dealer,Automatic,First Owner,1036,7,High,Low,0.0,Petrol_Automatic,Maruti,73.58,0
1036,CAR_001037,Hyundai i20 1.4 Magna Executive,2013,280000.0,52000.0,Diesel,Dealer,Manual,First Owner,1037,12,Low,Medium,0.0,Diesel_Manual,Hyundai,5.38,1
1037,CAR_001038,Maruti Baleno Zeta,2016,725000.0,49217.0,Petrol,Trustmark Dealer,Manual,First Owner,1038,9,High,Medium,0.0,Petrol_Manual,Maruti,14.73,0
1038,CAR_001039,Mahindra Verito 1.5 D4 BSIV,2015,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,1039,10,Low,High,0.0,Diesel_Manual,Mahindra,3.33,1
1039,CAR_001040,Hyundai Xcent 1.2 Kappa S,2015,375000.0,70000.0,Petrol,Individual,Manual,First Owner,1040,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,5.36,0
1040,CAR_001041,Honda City VTEC,2003,310000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,1041,22,Mid,High,0.0,Petrol_Manual,Honda,3.44,0
1041,CAR_001042,Maruti Swift Dzire VDi,2011,375000.0,190000.0,Diesel,Individual,Manual,Third Owner,1042,14,Mid,Medium,0.0,Diesel_Manual,Maruti,1.97,1
1042,CAR_001043,Ford Endeavour 3.2 Titanium AT 4X4,2016,1950000.0,106000.0,Diesel,Individual,Manual,First Owner,1043,9,Premium,High,0.0,Diesel_Automatic,Ford,18.4,1
1043,CAR_001044,Tata Xenon XT EX 4X2,2012,400000.0,80000.0,Diesel,Individual,Manual,Second Owner,1044,13,Mid,High,0.0,Diesel_Manual,Tata,5.0,1
1044,CAR_001045,Nissan Terrano XL Plus 85 PS,2014,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,1045,11,Mid,High,0.0,Diesel_Manual,Nissan,6.88,1
1045,CAR_001046,Maruti Eeco 7 Seater Standard BSIV,2018,270000.0,20000.0,Petrol,Individual,Manual,First Owner,1046,7,Low,Medium,0.0,Petrol_Manual,Maruti,13.5,0
1046,CAR_001047,Maruti Ertiga ZDI,2012,555000.0,110000.0,Diesel,Individual,Manual,Second Owner,1047,13,Mid,High,0.0,Diesel_Manual,Maruti,5.05,1
1047,CAR_001048,Ford Ecosport 1.5 DV5 MT Ambiente,2013,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,1048,12,Mid,Medium,0.0,Diesel_Manual,Ford,8.33,1
1048,CAR_001049,Skoda Laura Elegance 1.9 TDI  AT,2010,400000.0,130000.0,Diesel,Individual,Automatic,Second Owner,1049,15,Mid,High,0.0,Diesel_Automatic,Skoda,3.08,1
1049,CAR_001050,Hyundai EON 1.0 Era Plus,2017,280000.0,28838.0,Petrol,Individual,Manual,First Owner,1050,8,Low,Low,0.0,Petrol_Manual,Hyundai,9.71,0
1050,CAR_001051,Tata Indica GLS BS IV,2008,80000.0,200000.0,Petrol,Individual,Manual,First Owner,1051,17,Low,Very High,0.0,Petrol_Manual,Tata,0.4,0
1051,CAR_001052,Tata Indica DLS,2006,55000.0,100000.0,Diesel,Individual,Manual,First Owner,1052,19,Low,High,0.0,Diesel_Manual,Tata,0.55,1
1052,CAR_001053,Maruti Alto 800 LXI Airbag,2014,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,1053,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
1053,CAR_001054,Mahindra Bolero SLX 2WD,2010,250000.0,190000.0,Petrol,Individual,Manual,Fourth & Above Owner,1054,15,Low,Very High,0.0,Diesel_Manual,Mahindra,1.32,1
1054,CAR_001055,Mahindra Jeep MM 540,2007,95000.0,100000.0,Diesel,Individual,Manual,Second Owner,1055,18,Low,High,0.0,Diesel_Manual,Mahindra,0.95,1
1055,CAR_001056,Mahindra Scorpio 1.99 S10,2014,4461000.0,135000.0,Diesel,Individual,Manual,Second Owner,1056,11,High,High,0.0,Diesel_Manual,Mahindra,5.37,1
1056,CAR_001057,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,4461000.0,50000.0,Diesel,Individual,Manual,Third Owner,1057,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,6.4,1
1057,CAR_001058,Skoda Octavia Elegance 2.0 TDI AT,2014,4461000.0,135000.0,Diesel,Individual,Automatic,Third Owner,1058,11,Premium,High,0.0,Diesel_Automatic,Skoda,8.89,1
1058,CAR_001059,Chevrolet Beat Diesel LS,2012,160000.0,50000.0,Diesel,Individual,Manual,First Owner,1059,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,3.2,1
1059,CAR_001060,Honda Jazz 1.5 VX i DTEC,2018,790000.0,19571.0,Diesel,Dealer,Manual,First Owner,1060,7,High,Low,0.0,Diesel_Manual,Honda,40.37,1
1060,CAR_001061,Honda City i-VTEC ZX,2018,1200000.0,29600.0,Petrol,Dealer,Manual,First Owner,1061,7,Premium,Low,0.0,Petrol_Manual,Honda,40.54,0
1061,CAR_001062,Maruti Swift LXI,2011,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,1062,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
1062,CAR_001063,Maruti Vitara Brezza ZDi Plus AMT Dual Tone,2018,950000.0,60000.0,Diesel,Individual,Automatic,First Owner,1063,7,High,Low,0.0,Diesel_Automatic,Maruti,63.33,1
1063,CAR_001064,Maruti Celerio LXI MT BSIV,2019,340000.0,50000.0,Petrol,Individual,Manual,First Owner,1064,6,Mid,Medium,0.0,Petrol_Manual,Maruti,6.8,0
1064,CAR_001065,Renault Captur 1.5 Diesel RXT,2017,825000.0,13500.0,Diesel,Individual,Manual,First Owner,1065,8,High,Low,0.0,Diesel_Manual,Renault,61.11,1
1065,CAR_001066,Audi A4 30 TFSI Technology,2018,3100000.0,22000.0,Petrol,Individual,Automatic,First Owner,1066,7,Premium,Low,0.0,Petrol_Automatic,Audi,140.91,0
1066,CAR_001067,Honda Amaze VX Diesel BSIV,2018,780000.0,25000.0,Diesel,Dealer,Manual,First Owner,1067,7,High,Low,0.0,Diesel_Manual,Honda,31.2,1
1067,CAR_001068,Toyota Corolla Altis 1.8 VL AT,2010,350000.0,80000.0,Petrol,Individual,Automatic,Third Owner,1068,15,Mid,High,0.0,Petrol_Automatic,Toyota,4.38,0
1068,CAR_001069,Honda Amaze VX Petrol BSIV,2018,690000.0,39000.0,Petrol,Dealer,Manual,First Owner,1069,7,High,Medium,0.0,Petrol_Manual,Honda,17.69,0
1069,CAR_001070,Maruti Alto 800 VXI,2016,245000.0,60000.0,Petrol,Individual,Manual,First Owner,1070,9,Low,Medium,0.0,Petrol_Manual,Maruti,4.08,0
1070,CAR_001071,Honda Amaze VX Diesel BSIV,2018,790000.0,60000.0,Diesel,Dealer,Manual,First Owner,1071,7,High,Medium,0.0,Diesel_Manual,Honda,16.12,1
1071,CAR_001072,Honda Amaze VX i-VTEC,2018,680000.0,48600.0,Petrol,Dealer,Manual,First Owner,1072,7,High,Medium,0.0,Petrol_Manual,Honda,13.99,0
1072,CAR_001073,Hyundai Grand i10 Asta Option,2017,540000.0,20000.0,Diesel,Individual,Manual,Second Owner,1073,8,Mid,Low,0.0,Petrol_Manual,Hyundai,27.0,0
1073,CAR_001074,Mahindra KUV 100 mFALCON G80 K2,2016,425000.0,25000.0,Petrol,Individual,Manual,First Owner,1074,9,Mid,Low,0.0,Petrol_Manual,Mahindra,17.0,0
1074,CAR_001075,Maruti Zen Estilo Sports,2008,140000.0,120000.0,Petrol,Individual,Manual,First Owner,1075,17,Low,High,0.0,Petrol_Manual,Maruti,1.17,0
1075,CAR_001076,Maruti Ertiga ZDI Plus,2019,1100000.0,60000.0,Diesel,Individual,Manual,First Owner,1076,6,Premium,Low,0.0,Diesel_Manual,Maruti,36.67,1
1076,CAR_001077,Renault Lodgy 85PS RxL,2015,715000.0,35000.0,Diesel,Individual,Manual,First Owner,1077,10,High,Medium,0.0,Diesel_Manual,Renault,20.43,1
1077,CAR_001078,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,1078,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
1078,CAR_001079,Audi Q3 2.0 TDI Quattro Premium Plus,2015,1750000.0,127643.0,Diesel,Dealer,Automatic,First Owner,1079,10,Premium,High,0.0,Diesel_Automatic,Audi,13.71,1
1079,CAR_001080,Tata Zest Revotron 1.2T XMS,2018,400000.0,54000.0,Petrol,Dealer,Manual,Second Owner,1080,7,Mid,Medium,0.0,Petrol_Manual,Tata,7.41,0
1080,CAR_001081,Audi A6 2.0 TDI  Design Edition,2014,1750000.0,102354.0,Diesel,Dealer,Automatic,First Owner,1081,11,Premium,High,0.0,Diesel_Automatic,Audi,17.1,1
1081,CAR_001082,Audi Q5 2.0 TFSI Quattro Premium Plus,2014,1850000.0,62237.0,CNG,Dealer,Automatic,First Owner,1082,11,Premium,Medium,0.0,Petrol_Automatic,Audi,29.73,0
1082,CAR_001083,Hyundai Santro Xing XL eRLX Euro III,2005,114999.0,90000.0,Petrol,Dealer,Manual,Second Owner,1083,20,Low,High,0.0,Petrol_Manual,Hyundai,1.28,0
1083,CAR_001084,Ford EcoSport 1.5 TDCi Titanium BSIV,2018,950000.0,21394.0,Diesel,Dealer,Manual,First Owner,1084,7,High,Low,0.0,Diesel_Manual,Ford,44.4,1
1084,CAR_001085,Datsun GO T Petrol,2015,310000.0,32686.0,Petrol,Dealer,Manual,First Owner,1085,10,Low,Medium,0.0,Petrol_Manual,Datsun,9.48,0
1085,CAR_001086,Mahindra Jeep CJ 500 DI,2006,575000.0,1001.0,LPG,Dealer,Manual,First Owner,1086,19,Mid,Low,0.0,Diesel_Manual,Mahindra,574.43,1
1086,CAR_001087,Renault Pulse RxZ,2017,490000.0,22000.0,Diesel,Dealer,Manual,First Owner,1087,8,Mid,Low,0.0,Diesel_Manual,Renault,22.27,1
1087,CAR_001088,Hyundai Grand i10 Magna,2017,480000.0,53261.0,Petrol,Dealer,Manual,First Owner,1088,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.01,0
1088,CAR_001089,Land Rover Discovery Sport SD4 HSE Luxury,2016,3500000.0,53000.0,Diesel,Dealer,Automatic,First Owner,1089,9,Low,Medium,0.0,Diesel_Automatic,Land,66.04,1
1089,CAR_001090,Renault KWID 1.0 RXT Optional,2018,370000.0,10000.0,Petrol,Dealer,Manual,First Owner,1090,7,Mid,Low,0.0,Petrol_Manual,Renault,37.0,0
1090,CAR_001091,Maruti Baleno Alpha 1.3,2017,835000.0,60000.0,Diesel,Dealer,Manual,First Owner,1091,8,Low,Medium,0.0,Diesel_Manual,Maruti,59.64,1
1091,CAR_001092,Volkswagen Polo 2015-2019 1.2 MPI Highline,2017,650000.0,44000.0,Petrol,Dealer,Manual,First Owner,1092,8,High,Medium,0.0,Petrol_Manual,Volkswagen,14.77,0
1092,CAR_001093,Toyota Etios VD,2014,670000.0,39895.0,Diesel,Dealer,Manual,First Owner,1093,11,Low,Medium,0.0,Diesel_Manual,Toyota,16.79,1
1093,CAR_001094,Chevrolet Cruze LTZ AT,2017,950000.0,73000.0,Diesel,Dealer,Automatic,First Owner,1094,8,Low,High,0.0,Diesel_Automatic,Chevrolet,13.01,1
1094,CAR_001095,Honda City i-VTEC SV,2014,690000.0,17000.0,Petrol,Dealer,Manual,First Owner,1095,11,High,Low,0.0,Petrol_Manual,Honda,40.59,0
1095,CAR_001096,Volkswagen Polo 2015-2019 1.2 MPI Highline,2017,670000.0,18591.0,Petrol,Dealer,Manual,First Owner,1096,8,High,Low,0.0,Petrol_Manual,Volkswagen,36.04,0
1096,CAR_001097,Tata Nexon 1.5 Revotorq XZ,2017,925000.0,26766.0,Diesel,Dealer,Manual,First Owner,1097,8,Low,Low,0.0,Diesel_Manual,Tata,34.56,1
1097,CAR_001098,Hyundai EON Magna Plus,2015,250000.0,30000.0,Petrol,Individual,Manual,First Owner,1098,10,Low,Low,0.0,Petrol_Manual,Hyundai,8.33,0
1098,CAR_001099,Tata Sumo Gold EX,2016,509999.0,42000.0,Diesel,Individual,Manual,First Owner,1099,9,Mid,Medium,0.0,Diesel_Manual,Tata,12.14,1
1099,CAR_001100,Honda City i DTEC VX,2014,520000.0,110000.0,Diesel,Individual,Manual,First Owner,1100,11,Mid,Medium,0.0,Diesel_Manual,Honda,4.73,1
1100,CAR_001101,Tata Indica DLE,2005,100000.0,60000.0,Diesel,Individual,Manual,First Owner,1101,20,Low,High,0.0,Diesel_Manual,Tata,1.0,1
1101,CAR_001102,Tata Indica DLS,2006,85000.0,300000.0,Electric,Individual,Manual,Second Owner,1102,19,Low,Very High,0.0,Diesel_Manual,Tata,0.28,1
1102,CAR_001103,Tata Indigo LX,2009,130000.0,110000.0,Diesel,Individual,Manual,Third Owner,1103,16,Low,High,0.0,Diesel_Manual,Tata,1.18,1
1103,CAR_001104,Ford EcoSport 1.5 TDCi Titanium BSIV,2018,950000.0,27620.0,Diesel,Dealer,Manual,First Owner,1104,7,High,Low,0.0,Diesel_Manual,Ford,34.4,1
1104,CAR_001105,Ford Figo Aspire 1.2 Ti-VCT Titanium Plus,2015,450000.0,70000.0,Petrol,Individual,Manual,First Owner,1105,10,Low,Medium,0.0,Petrol_Manual,Ford,6.43,0
1105,CAR_001106,Honda City S,2012,320000.0,50000.0,Petrol,Individual,Manual,Second Owner,1106,13,Mid,Medium,0.0,Petrol_Manual,Honda,6.4,0
1106,CAR_001107,Maruti Swift Dzire VDI,2015,480000.0,50000.0,Diesel,Individual,Manual,First Owner,1107,10,Mid,Medium,0.0,Diesel_Manual,Maruti,9.6,1
1107,CAR_001108,Mercedes-Benz New C-Class C 220 CDI Grand Edition,2014,2490000.0,46000.0,Diesel,Individual,Automatic,First Owner,1108,11,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,54.13,1
1108,CAR_001109,Toyota Innova 2.5 VX 8 STR,2012,790000.0,82000.0,Diesel,Individual,Manual,First Owner,1109,13,High,High,0.0,Diesel_Manual,Toyota,9.63,1
1109,CAR_001110,Maruti Omni MPI STD BSIV,2016,120000.0,120000.0,Petrol,Individual,Manual,First Owner,1110,9,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
1110,CAR_001111,Maruti Alto LXi,2006,75000.0,60000.0,Petrol,Individual,Manual,Third Owner,1111,19,Low,High,0.0,Petrol_Manual,Maruti,0.94,0
1111,CAR_001112,Tata Indica GLS BS IV,2010,99000.0,70000.0,Petrol,Individual,Manual,First Owner,1112,15,Low,Medium,0.0,Petrol_Manual,Tata,1.41,0
1112,CAR_001113,Maruti S-Cross Zeta DDiS 200 SH,2018,1015000.0,20000.0,Diesel,Individual,Manual,First Owner,1113,7,Premium,Medium,0.0,Diesel_Manual,Maruti,50.75,1
1113,CAR_001114,Mahindra Xylo D4 BSIV,2013,250000.0,200000.0,Diesel,Individual,Manual,Second Owner,1114,12,Low,Very High,0.0,Diesel_Manual,Mahindra,1.25,1
1114,CAR_001115,Toyota Innova Crysta 2.5 VX BS IV,2011,520000.0,120000.0,Petrol,Individual,Manual,First Owner,1115,14,Mid,High,0.0,Diesel_Manual,Toyota,4.33,1
1115,CAR_001116,Hyundai Verna CRDi 1.6 AT SX Option,2018,1100000.0,30000.0,Diesel,Individual,Automatic,First Owner,1116,7,Premium,Low,0.0,Diesel_Automatic,Hyundai,36.67,1
1116,CAR_001117,Toyota Innova 2.5 V Diesel 7-seater,2005,4461000.0,223000.0,Diesel,Individual,Manual,First Owner,1117,20,Low,Very High,0.0,Diesel_Manual,Toyota,0.9,1
1117,CAR_001118,Maruti Swift Dzire VDI,2016,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,1118,9,Mid,Medium,0.0,Diesel_Manual,Maruti,8.75,1
1118,CAR_001119,Hyundai i20 Asta 1.2,2016,500000.0,40000.0,Petrol,Individual,Manual,First Owner,1119,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,12.5,0
1119,CAR_001120,Mahindra Scorpio S2 7 Seater,2016,675000.0,161327.0,Diesel,Individual,Manual,First Owner,1120,9,High,Very High,0.0,Diesel_Manual,Mahindra,4.18,1
1120,CAR_001121,Hyundai i10 Magna 1.2 iTech SE,2009,150000.0,60000.0,Petrol,Individual,Manual,Third Owner,1121,16,Low,Medium,0.0,Petrol_Manual,Hyundai,2.5,0
1121,CAR_001122,Mahindra Bolero DI,2007,250000.0,50000.0,Diesel,Individual,Manual,First Owner,1122,18,Low,Medium,0.0,Diesel_Manual,Mahindra,5.0,1
1122,CAR_001123,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2018,450000.0,6000.0,Petrol,Individual,Manual,First Owner,1123,7,Low,Low,0.0,Petrol_Manual,Hyundai,75.0,0
1123,CAR_001124,Hyundai Santro Xing XS,2009,150000.0,68000.0,Petrol,Individual,Manual,Second Owner,1124,16,Low,Medium,0.0,Petrol_Manual,Hyundai,2.21,0
1124,CAR_001125,Chevrolet Beat Diesel LS,2012,200000.0,85000.0,Diesel,Individual,Manual,First Owner,1125,13,Low,High,0.0,Diesel_Manual,Chevrolet,2.35,1
1125,CAR_001126,Mahindra Scorpio VLS AT 2.2 mHAWK,2011,525000.0,70000.0,Diesel,Individual,Automatic,First Owner,1126,14,Mid,Medium,0.0,Diesel_Automatic,Mahindra,7.5,1
1126,CAR_001127,Maruti Esteem VX,1999,60000.0,35000.0,Petrol,Individual,Manual,Second Owner,1127,26,Low,Medium,0.0,Petrol_Manual,Maruti,1.71,0
1127,CAR_001128,Maruti Swift Vdi BSIII,2010,180000.0,100000.0,Diesel,Individual,Manual,Second Owner,1128,15,Low,High,0.0,Diesel_Manual,Maruti,1.8,1
1128,CAR_001129,Honda Amaze S i-Dtech,2015,475000.0,110000.0,Diesel,Individual,Manual,First Owner,1129,10,Mid,High,0.0,Diesel_Manual,Honda,4.32,1
1129,CAR_001130,Mahindra Scorpio VLS AT 2.2 mHAWK,2010,380000.0,195000.0,Diesel,Individual,Automatic,Second Owner,1130,15,Low,Very High,0.0,Diesel_Automatic,Mahindra,1.95,1
1130,CAR_001131,Maruti Alto 800 LXI,2015,285000.0,60000.0,Petrol,Individual,Manual,Second Owner,1131,10,Low,Medium,0.0,Petrol_Manual,Maruti,6.2,0
1131,CAR_001132,Maruti Alto LXi BSIII,2008,91200.0,60000.0,Diesel,Individual,Manual,Second Owner,1132,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.52,0
1132,CAR_001133,Datsun RediGO T Option,2017,250000.0,60000.0,Petrol,Individual,Manual,First Owner,1133,8,Low,Medium,0.0,Petrol_Manual,Datsun,5.43,0
1133,CAR_001134,Hyundai Grand i10 CRDi Sportz,2015,400000.0,42000.0,Diesel,Individual,Manual,First Owner,1134,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,9.52,1
1134,CAR_001135,Hyundai Santro Xing XG AT,2006,160000.0,90000.0,Petrol,Individual,Automatic,Second Owner,1135,19,Low,Medium,0.0,Petrol_Automatic,Hyundai,1.78,0
1135,CAR_001136,Skoda Yeti Ambition 4X2,2012,650000.0,85000.0,Diesel,Individual,Manual,Second Owner,1136,13,High,High,0.0,Diesel_Manual,Skoda,7.65,1
1136,CAR_001137,Hyundai Verna 1.6 SX,2012,409999.0,120000.0,Diesel,Individual,Manual,Third Owner,1137,13,Mid,High,0.0,Diesel_Manual,Hyundai,3.42,1
1137,CAR_001138,Ford Ecosport 1.5 DV5 MT Titanium,2013,425000.0,72000.0,Diesel,Individual,Manual,First Owner,1138,12,Mid,Medium,0.0,Diesel_Manual,Ford,5.9,1
1138,CAR_001139,Maruti Wagon R LXI CNG,2013,250000.0,60000.0,CNG,Individual,Manual,Second Owner,1139,12,Low,High,0.0,CNG_Manual,Maruti,3.52,0
1139,CAR_001140,Maruti Swift Dzire VXi,2009,200000.0,110000.0,Petrol,Individual,Manual,Fourth & Above Owner,1140,16,Low,High,0.0,Petrol_Manual,Maruti,1.82,0
1140,CAR_001141,Maruti Wagon R LXI CNG,2013,250000.0,71000.0,CNG,Individual,Manual,Second Owner,1141,12,Low,Medium,0.0,CNG_Manual,Maruti,3.52,0
1141,CAR_001142,Volkswagen Polo Diesel Trendline 1.2L,2010,130000.0,144000.0,Diesel,Individual,Manual,Fourth & Above Owner,1142,15,Low,High,0.0,Diesel_Manual,Volkswagen,0.9,1
1142,CAR_001143,Maruti Eeco 5 Seater AC BSIV,2017,260000.0,60000.0,CNG,Dealer,Manual,First Owner,1143,8,Low,Low,0.0,Petrol_Manual,Maruti,8.67,0
1143,CAR_001144,Hyundai Verna SX,2007,155000.0,65000.0,Petrol,Dealer,Manual,First Owner,1144,18,Low,Medium,0.0,Petrol_Manual,Hyundai,2.38,0
1144,CAR_001145,Honda Jazz 1.5 VX i DTEC,2015,450000.0,38000.0,Diesel,Individual,Manual,First Owner,1145,10,Low,Medium,0.0,Diesel_Manual,Honda,11.84,1
1145,CAR_001146,Maruti SX4 Celebration Petrol,2012,220000.0,90000.0,Petrol,Individual,Manual,Second Owner,1146,13,Low,High,0.0,Petrol_Manual,Maruti,2.44,0
1146,CAR_001147,Hyundai i20 Active S Diesel,2018,4461000.0,37000.0,Diesel,Dealer,Manual,First Owner,1147,7,High,Medium,0.0,Diesel_Manual,Hyundai,17.57,1
1147,CAR_001148,Maruti Alto 800 LXI,2018,285000.0,30000.0,Petrol,Dealer,Manual,First Owner,1148,7,Low,Low,0.0,Petrol_Manual,Maruti,9.5,0
1148,CAR_001149,Honda Amaze V CVT Petrol BSIV,2018,725000.0,26000.0,Petrol,Dealer,Manual,First Owner,1149,7,High,Low,0.0,Petrol_Automatic,Honda,27.88,0
1149,CAR_001150,Maruti Ciaz ZXi,2016,4461000.0,30000.0,Petrol,Dealer,Manual,First Owner,1150,9,High,Low,0.0,Petrol_Manual,Maruti,20.83,0
1150,CAR_001151,Hyundai i20 Active SX Petrol,2016,550000.0,23000.0,Petrol,Dealer,Manual,First Owner,1151,9,Mid,Low,0.0,Petrol_Manual,Hyundai,23.91,0
1151,CAR_001152,Hyundai i10 Magna,2012,240000.0,40000.0,Petrol,Dealer,Manual,First Owner,1152,13,Low,Medium,0.0,Petrol_Manual,Hyundai,6.0,0
1152,CAR_001153,Nissan Terrano XL 85 PS,2014,600000.0,27000.0,Diesel,Dealer,Manual,First Owner,1153,11,Low,Low,0.0,Diesel_Manual,Nissan,22.22,1
1153,CAR_001154,Tata New Safari DICOR 2.2 EX 4x2,2012,350000.0,65000.0,LPG,Dealer,Manual,First Owner,1154,13,Mid,Medium,0.0,Diesel_Manual,Tata,5.38,1
1154,CAR_001155,Hyundai EON Magna Plus,2015,245000.0,32000.0,Petrol,Dealer,Manual,First Owner,1155,10,Low,Medium,0.0,Petrol_Manual,Hyundai,7.66,0
1155,CAR_001156,Mahindra Bolero B6,2016,600000.0,56000.0,Diesel,Dealer,Manual,First Owner,1156,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.71,1
1156,CAR_001157,BMW X1 sDrive 20d xLine,2017,2400000.0,30000.0,Diesel,Individual,Manual,First Owner,1157,8,Premium,Low,0.0,Diesel_Automatic,BMW,80.0,1
1157,CAR_001158,Mahindra Scorpio SLE BS IV,2014,4461000.0,80000.0,Diesel,Dealer,Manual,First Owner,1158,11,High,High,0.0,Diesel_Manual,Mahindra,7.75,1
1158,CAR_001159,Maruti Swift Dzire 1.2 Vxi BSIV,2014,395000.0,52000.0,Petrol,Dealer,Manual,Second Owner,1159,11,Low,Medium,0.0,Petrol_Manual,Maruti,7.6,0
1159,CAR_001160,Maruti Ertiga ZXI,2014,635000.0,13250.0,Petrol,Dealer,Manual,First Owner,1160,11,High,Low,0.0,Petrol_Manual,Maruti,47.92,0
1160,CAR_001161,Maruti Alto LX,2011,180000.0,36000.0,Petrol,Dealer,Manual,First Owner,1161,14,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
1161,CAR_001162,Maruti Swift ZDI Plus,2015,484999.0,73000.0,Diesel,Dealer,Manual,Second Owner,1162,10,Mid,High,0.0,Diesel_Manual,Maruti,6.64,1
1162,CAR_001163,Maruti Alto LX,2004,110000.0,60000.0,Petrol,Individual,Manual,First Owner,1163,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.83,0
1163,CAR_001164,Maruti Wagon R VXI BSIII,2006,150000.0,60000.0,Petrol,Individual,Manual,Second Owner,1164,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
1164,CAR_001165,Chevrolet Tavera Neo 2 LT L 9 Str,2012,400000.0,110000.0,Diesel,Individual,Manual,First Owner,1165,13,Mid,High,0.0,Diesel_Manual,Chevrolet,3.64,1
1165,CAR_001166,Tata Indica Vista Aura Plus 1.3 Quadrajet,2012,200000.0,60000.0,Diesel,Individual,Manual,Second Owner,1166,13,Low,Medium,0.0,Diesel_Manual,Tata,2.86,1
1166,CAR_001167,Renault Duster 85PS Diesel RxL Plus,2013,302000.0,60000.0,Electric,Individual,Manual,Second Owner,1167,12,Mid,Medium,0.0,Diesel_Manual,Renault,5.03,1
1167,CAR_001168,Maruti Swift Dzire Vdi BSIV,2011,215000.0,90000.0,Diesel,Individual,Manual,Fourth & Above Owner,1168,14,Low,Medium,0.0,Diesel_Manual,Maruti,2.39,1
1168,CAR_001169,Maruti Wagon R VXI BSIII,2004,155000.0,80000.0,Petrol,Individual,Manual,Second Owner,1169,21,Low,High,0.0,Petrol_Manual,Maruti,1.94,0
1169,CAR_001170,Toyota Innova 2.5 G (Diesel) 8 Seater,2014,1000000.0,101000.0,Diesel,Individual,Manual,Second Owner,1170,11,High,High,0.0,Diesel_Manual,Toyota,9.9,1
1170,CAR_001171,Mahindra Bolero Power Plus SLX,2018,4461000.0,44000.0,Diesel,Individual,Manual,First Owner,1171,7,Mid,Medium,0.0,Diesel_Manual,Mahindra,12.5,1
1171,CAR_001172,Maruti Swift 1.3 VXi,2006,200000.0,60000.0,Petrol,Individual,Manual,First Owner,1172,19,Low,Medium,0.0,Petrol_Manual,Maruti,4.44,0
1172,CAR_001173,Mahindra Scorpio S2 7 Seater,2017,750000.0,120000.0,Diesel,Individual,Manual,First Owner,1173,8,High,High,0.0,Diesel_Manual,Mahindra,6.25,1
1173,CAR_001174,Maruti Zen LXI,2005,90000.0,90000.0,CNG,Individual,Manual,Second Owner,1174,20,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
1174,CAR_001175,Maruti Wagon R LXI Minor,2007,80000.0,120000.0,Petrol,Individual,Manual,Second Owner,1175,18,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
1175,CAR_001176,Hyundai EON D Lite Plus,2013,229999.0,13500.0,LPG,Individual,Manual,First Owner,1176,12,Low,Low,0.0,Petrol_Manual,Hyundai,17.04,0
1176,CAR_001177,Renault KWID 1.0 RXT Optional,2017,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,1177,8,Mid,Low,0.0,Petrol_Manual,Renault,35.0,0
1177,CAR_001178,Hyundai Grand i10 Magna,2015,325000.0,25000.0,Petrol,Individual,Manual,Third Owner,1178,10,Mid,Low,0.0,Petrol_Manual,Hyundai,13.0,0
1178,CAR_001179,Tata Tigor 1.2 Revotron XT,2018,450000.0,8500.0,Petrol,Individual,Manual,First Owner,1179,7,Mid,Low,0.0,Petrol_Manual,Tata,52.94,0
1179,CAR_001180,Maruti SX4 Celebration Diesel,2012,250000.0,120000.0,Diesel,Individual,Manual,First Owner,1180,13,Low,High,0.0,Diesel_Manual,Maruti,2.08,1
1180,CAR_001181,Volkswagen Polo Diesel Comfortline 1.2L,2012,4461000.0,120000.0,Diesel,Individual,Manual,First Owner,1181,13,Low,Medium,0.0,Diesel_Manual,Volkswagen,1.88,1
1181,CAR_001182,Maruti Alto LXi,2009,150000.0,90246.0,Petrol,Individual,Manual,Third Owner,1182,16,Low,High,0.0,Petrol_Manual,Maruti,1.66,0
1182,CAR_001183,Maruti Alto LX BSIII,2008,110000.0,90000.0,Electric,Individual,Manual,Second Owner,1183,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.22,0
1183,CAR_001184,Maruti Alto LXi BSIII,2010,4461000.0,40000.0,Petrol,Individual,Manual,Second Owner,1184,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.75,0
1184,CAR_001185,Maruti Gypsy King Hard Top,2000,165000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,1185,25,Low,Medium,0.0,Petrol_Manual,Maruti,2.75,0
1185,CAR_001186,Renault Duster 85PS Diesel RxL,2017,610000.0,32000.0,Diesel,Individual,Manual,Second Owner,1186,8,Low,Medium,0.0,Diesel_Manual,Renault,19.06,1
1186,CAR_001187,Tata Manza Club Class Quadrajet90 VX,2013,170000.0,60400.0,Diesel,Individual,Manual,First Owner,1187,12,Low,Medium,0.0,Diesel_Manual,Tata,2.81,1
1187,CAR_001188,Hyundai Verna CRDi,2008,150000.0,70950.0,Diesel,Individual,Manual,Second Owner,1188,17,Low,High,0.0,Diesel_Manual,Hyundai,2.11,1
1188,CAR_001189,Hyundai i20 Asta 1.4 CRDi,2013,4461000.0,90000.0,Diesel,Individual,Manual,First Owner,1189,12,Low,High,0.0,Diesel_Manual,Hyundai,3.33,1
1189,CAR_001190,Hyundai Xcent 1.1 CRDi Base,2015,375000.0,72000.0,Diesel,Individual,Manual,First Owner,1190,10,Low,High,0.0,Diesel_Manual,Hyundai,5.21,1
1190,CAR_001191,Maruti Swift VDI,2013,390000.0,90000.0,Diesel,Individual,Manual,First Owner,1191,12,Mid,High,0.0,Diesel_Manual,Maruti,4.33,1
1191,CAR_001192,Hyundai i20 Active 1.2 S,2017,600000.0,10000.0,Petrol,Individual,Manual,First Owner,1192,8,Mid,Low,0.0,Petrol_Manual,Hyundai,60.0,0
1192,CAR_001193,Maruti Alto 800 LXI,2014,204999.0,25000.0,Petrol,Individual,Manual,Second Owner,1193,11,Low,Low,0.0,Petrol_Manual,Maruti,8.2,0
1193,CAR_001194,Hyundai Verna CRDi,2009,270000.0,80000.0,Diesel,Individual,Manual,Second Owner,1194,16,Low,High,0.0,Diesel_Manual,Hyundai,3.38,1
1194,CAR_001195,Hyundai Grand i10 Sportz,2017,375000.0,10000.0,Petrol,Individual,Manual,First Owner,1195,8,Mid,Low,0.0,Petrol_Manual,Hyundai,37.5,0
1195,CAR_001196,Maruti Baleno Zeta,2020,700000.0,1100.0,Petrol,Individual,Manual,First Owner,1196,5,High,Low,0.0,Petrol_Manual,Maruti,636.36,0
1196,CAR_001197,Maruti Swift VXI Deca,2017,455000.0,40000.0,Petrol,Individual,Manual,First Owner,1197,8,Mid,Medium,0.0,Petrol_Manual,Maruti,11.38,0
1197,CAR_001198,Hyundai i10 Sportz 1.1L,2017,375000.0,60000.0,Diesel,Individual,Manual,First Owner,1198,8,Mid,Low,0.0,Petrol_Manual,Hyundai,37.5,0
1198,CAR_001199,Hyundai Grand i10 Asta,2014,341000.0,31491.0,Petrol,Individual,Manual,First Owner,1199,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,10.83,0
1199,CAR_001200,Maruti Swift Dzire VDI,2014,200000.0,107143.0,Diesel,Individual,Manual,Second Owner,1200,11,Low,High,0.0,Diesel_Manual,Maruti,1.87,1
1200,CAR_001201,Maruti Ignis 1.2 AMT Delta BSIV,2017,4461000.0,15000.0,CNG,Individual,Automatic,First Owner,1201,8,Low,Low,0.0,Petrol_Automatic,Maruti,30.0,0
1201,CAR_001202,Hyundai Santro Xing GL Plus,2012,170000.0,50000.0,Petrol,Individual,Manual,Second Owner,1202,13,Low,Medium,0.0,Petrol_Manual,Hyundai,3.4,0
1202,CAR_001203,Toyota Innova Crysta 2.4 VX MT 8S BSIV,2017,1900000.0,60000.0,Diesel,Individual,Manual,First Owner,1203,8,Premium,High,0.0,Diesel_Manual,Toyota,19.0,1
1203,CAR_001204,Mahindra Bolero Power Plus LX,2016,500000.0,60000.0,Diesel,Individual,Manual,First Owner,1204,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.33,1
1204,CAR_001205,Maruti Vitara Brezza ZDi Plus,2016,4461000.0,85000.0,Diesel,Individual,Manual,First Owner,1205,9,High,High,0.0,Diesel_Manual,Maruti,9.65,1
1205,CAR_001206,Maruti Swift 1.2 DLX,2014,351000.0,40000.0,Petrol,Individual,Manual,Second Owner,1206,11,Mid,Medium,0.0,Petrol_Manual,Maruti,8.78,0
1206,CAR_001207,Hyundai i10 Magna 1.2 iTech SE,2011,195000.0,80000.0,Petrol,Individual,Manual,Second Owner,1207,14,Low,Medium,0.0,Petrol_Manual,Hyundai,2.44,0
1207,CAR_001208,Maruti Swift Dzire VDi,2008,4461000.0,82000.0,Diesel,Individual,Manual,Second Owner,1208,17,Low,High,0.0,Diesel_Manual,Maruti,3.05,1
1208,CAR_001209,Maruti Zen Estilo 1.1 VXI BSIII,2008,120000.0,70000.0,Petrol,Individual,Manual,Second Owner,1209,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.71,0
1209,CAR_001210,Maruti A-Star Vxi,2009,180000.0,70000.0,Petrol,Individual,Manual,Third Owner,1210,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.57,0
1210,CAR_001211,Mahindra KUV 100 mFALCON D75 K6,2016,430000.0,20000.0,Diesel,Individual,Manual,First Owner,1211,9,Mid,Low,0.0,Diesel_Manual,Mahindra,21.5,1
1211,CAR_001212,Tata Nano Cx BSIV,2012,75000.0,35000.0,Petrol,Individual,Manual,Second Owner,1212,13,Low,Medium,0.0,Petrol_Manual,Tata,2.14,0
1212,CAR_001213,Mahindra Xylo D4,2017,750000.0,80000.0,Diesel,Individual,Manual,First Owner,1213,8,High,High,0.0,Diesel_Manual,Mahindra,9.38,1
1213,CAR_001214,Tata Nano Lx,2011,60000.0,40000.0,LPG,Individual,Manual,First Owner,1214,14,Low,Medium,0.0,Petrol_Manual,Tata,1.5,0
1214,CAR_001215,Toyota Innova 2.5 VX (Diesel) 8 Seater,2015,1500000.0,46412.0,Diesel,Individual,Manual,First Owner,1215,10,Premium,Medium,0.0,Diesel_Manual,Toyota,32.32,1
1215,CAR_001216,Mahindra Bolero SLX 2WD BSIII,2011,390000.0,150000.0,Diesel,Individual,Manual,Fourth & Above Owner,1216,14,Mid,High,0.0,Diesel_Manual,Mahindra,2.6,1
1216,CAR_001217,Mahindra TUV 300 T4,2016,630000.0,50000.0,Diesel,Individual,Manual,First Owner,1217,9,High,Medium,0.0,Diesel_Manual,Mahindra,12.6,1
1217,CAR_001218,Tata Safari Storme VX Varicor 400,2018,1085000.0,15000.0,Diesel,Individual,Manual,First Owner,1218,7,Premium,Medium,0.0,Diesel_Manual,Tata,72.33,1
1218,CAR_001219,Hyundai i10 Magna 1.2 iTech SE,2012,190000.0,60000.0,Petrol,Individual,Manual,Second Owner,1219,13,Low,Medium,0.0,Petrol_Manual,Hyundai,3.17,0
1219,CAR_001220,Maruti Baleno Delta Automatic,2018,550000.0,25000.0,Petrol,Individual,Automatic,First Owner,1220,7,Mid,Low,0.0,Petrol_Automatic,Maruti,22.0,0
1220,CAR_001221,Maruti Alto 800 LXI,2012,200000.0,90000.0,Petrol,Individual,Manual,Third Owner,1221,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.22,0
1221,CAR_001222,Chevrolet Spark 1.0 LT,2011,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,1222,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.75,0
1222,CAR_001223,Ford Figo Diesel Celebration Edition,2013,260000.0,65000.0,Diesel,Individual,Manual,First Owner,1223,12,Low,Medium,0.0,Diesel_Manual,Ford,4.0,1
1223,CAR_001224,Maruti Alto LX,2011,151000.0,40000.0,Petrol,Individual,Manual,First Owner,1224,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.78,0
1224,CAR_001225,Maruti Vitara Brezza VDi,2018,780000.0,50000.0,Diesel,Individual,Manual,First Owner,1225,7,Low,Medium,0.0,Diesel_Manual,Maruti,15.6,1
1225,CAR_001226,Tata Indica Vista Quadrajet LX,2012,175000.0,80000.0,Diesel,Individual,Manual,Third Owner,1226,13,Low,High,0.0,Diesel_Manual,Tata,2.19,1
1226,CAR_001227,Tata Indica Vista TDI LS,2010,170000.0,107500.0,Diesel,Dealer,Manual,First Owner,1227,15,Low,High,0.0,Diesel_Manual,Tata,1.58,1
1227,CAR_001228,Tata Indica Vista TDI LS,2013,210000.0,135000.0,Diesel,Dealer,Manual,Second Owner,1228,12,Low,High,0.0,Diesel_Manual,Tata,1.56,1
1228,CAR_001229,Maruti Alto LXi,2007,90000.0,43826.0,Petrol,Dealer,Manual,Second Owner,1229,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.05,0
1229,CAR_001230,Hyundai i20 Active S Petrol,2016,535000.0,55838.0,Electric,Dealer,Manual,First Owner,1230,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.58,0
1230,CAR_001231,Maruti Swift VDI BSIV,2007,195000.0,112880.0,Diesel,Dealer,Manual,First Owner,1231,18,Low,High,0.0,Diesel_Manual,Maruti,1.73,1
1231,CAR_001232,Tata Indica Vista Terra Quadrajet 1.3L,2011,4461000.0,90000.0,Diesel,Individual,Manual,Third Owner,1232,14,Low,High,0.0,Diesel_Manual,Tata,1.56,1
1232,CAR_001233,Nissan Sunny XL,2012,350000.0,40000.0,Petrol,Individual,Manual,First Owner,1233,13,Mid,Medium,0.0,Petrol_Manual,Nissan,8.75,0
1233,CAR_001234,Hyundai Verna 1.6 SX VTVT (O),2013,425000.0,60000.0,Petrol,Individual,Manual,First Owner,1234,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.44,0
1234,CAR_001235,Renault KWID 1.0 RXT Optional,2017,300000.0,30300.0,Petrol,Individual,Manual,First Owner,1235,8,Low,Medium,0.0,Petrol_Manual,Renault,9.9,0
1235,CAR_001236,Volkswagen Vento Diesel Style Limited Edition,2013,4461000.0,85000.0,Petrol,Individual,Manual,Second Owner,1236,12,Mid,High,0.0,Diesel_Manual,Volkswagen,4.0,1
1236,CAR_001237,Maruti Swift Dzire VXi,2009,250000.0,80659.0,Petrol,Dealer,Manual,First Owner,1237,16,Low,High,0.0,Petrol_Manual,Maruti,3.1,0
1237,CAR_001238,Chevrolet Aveo 1.4 LS,2008,170000.0,60000.0,Petrol,Dealer,Manual,First Owner,1238,17,Low,High,0.0,Petrol_Manual,Chevrolet,2.09,0
1238,CAR_001239,Honda Amaze S i-Dtech,2013,315000.0,127884.0,Diesel,Dealer,Manual,First Owner,1239,12,Mid,High,0.0,Diesel_Manual,Honda,2.46,1
1239,CAR_001240,Chevrolet Beat PS,2015,200000.0,50000.0,Petrol,Individual,Manual,First Owner,1240,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,4.0,0
1240,CAR_001241,Chevrolet Beat PS,2016,300000.0,25000.0,Petrol,Individual,Manual,First Owner,1241,9,Low,Low,0.0,Petrol_Manual,Chevrolet,12.0,0
1241,CAR_001242,Honda Amaze S i-Dtech,2015,300000.0,66755.0,Diesel,Dealer,Manual,First Owner,1242,10,Low,Medium,0.0,Diesel_Manual,Honda,4.49,1
1242,CAR_001243,Honda City E,2013,4461000.0,123084.0,Petrol,Dealer,Manual,First Owner,1243,12,Mid,High,0.0,Petrol_Manual,Honda,3.09,0
1243,CAR_001244,Maruti Swift VXI BSIII,2009,250000.0,806599.0,CNG,Dealer,Manual,First Owner,1244,16,Low,Very High,0.0,Petrol_Manual,Maruti,0.31,0
1244,CAR_001245,Honda Amaze S i-Dtech,2013,310000.0,95851.0,Diesel,Dealer,Manual,First Owner,1245,12,Mid,Medium,0.0,Diesel_Manual,Honda,3.23,1
1245,CAR_001246,Maruti Wagon R VXI Minor ABS,2010,260000.0,50000.0,Petrol,Individual,Manual,First Owner,1246,15,Low,Medium,0.0,Petrol_Manual,Maruti,5.2,0
1246,CAR_001247,Maruti Omni BSIII 8-STR W/ IMMOBILISER,2008,150000.0,60000.0,Petrol,Individual,Manual,Second Owner,1247,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
1247,CAR_001248,Maruti Esteem Vxi - BSIII,2006,85000.0,60000.0,Petrol,Individual,Manual,Second Owner,1248,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.42,0
1248,CAR_001249,Maruti Alto 800 LXI,2015,280000.0,15000.0,Petrol,Individual,Manual,Second Owner,1249,10,Low,Low,0.0,Petrol_Manual,Maruti,18.67,0
1249,CAR_001250,Mahindra Xylo H4 ABS,2016,850000.0,40000.0,Diesel,Individual,Manual,First Owner,1250,9,High,Medium,0.0,Diesel_Manual,Mahindra,21.25,1
1250,CAR_001251,Datsun GO Plus T Option BSIV,2017,375000.0,60000.0,Petrol,Individual,Manual,Second Owner,1251,8,Mid,Low,0.0,Petrol_Manual,Datsun,18.75,0
1251,CAR_001252,Hyundai Elantra CRDi (Leather Option),2006,125000.0,120000.0,LPG,Individual,Manual,Third Owner,1252,19,Low,High,0.0,Diesel_Manual,Hyundai,1.04,1
1252,CAR_001253,Chevrolet Beat Diesel LT,2012,250000.0,110000.0,Diesel,Individual,Manual,First Owner,1253,13,Low,High,0.0,Diesel_Manual,Chevrolet,2.27,1
1253,CAR_001254,Toyota Corolla Altis D-4D J,2014,715000.0,234000.0,Diesel,Individual,Manual,First Owner,1254,11,High,Medium,0.0,Diesel_Manual,Toyota,3.06,1
1254,CAR_001255,Hyundai i20 1.4 CRDi Era,2011,220000.0,60000.0,Diesel,Individual,Manual,First Owner,1255,14,Low,Medium,0.0,Diesel_Manual,Hyundai,3.14,1
1255,CAR_001256,Tata Indica Vista Aqua 1.3 Quadrajet,2010,110000.0,110000.0,Diesel,Individual,Manual,Second Owner,1256,15,Low,High,0.0,Diesel_Manual,Tata,1.0,1
1256,CAR_001257,Hyundai Verna XXi (Petrol),2007,150000.0,84000.0,Petrol,Individual,Manual,Second Owner,1257,18,Low,High,0.0,Petrol_Manual,Hyundai,1.79,0
1257,CAR_001258,Nissan Sunny Diesel XV,2012,4461000.0,170000.0,Diesel,Individual,Manual,First Owner,1258,13,Low,Very High,0.0,Diesel_Manual,Nissan,1.76,1
1258,CAR_001259,Datsun RediGO T Option,2017,220000.0,40000.0,Petrol,Individual,Manual,First Owner,1259,8,Low,Medium,0.0,Petrol_Manual,Datsun,5.5,0
1259,CAR_001260,Maruti Wagon R LX,2006,75000.0,96000.0,Petrol,Individual,Manual,Second Owner,1260,19,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
1260,CAR_001261,Tata Indigo LS,2009,100000.0,20000.0,Diesel,Individual,Manual,First Owner,1261,16,Low,Low,0.0,Diesel_Manual,Tata,5.0,1
1261,CAR_001262,Ford Figo Aspire 1.2 Ti-VCT Trend,2016,380000.0,15000.0,Electric,Individual,Manual,First Owner,1262,9,Mid,Low,0.0,Petrol_Manual,Ford,25.33,0
1262,CAR_001263,Hyundai Santro Xing XK (Non-AC),2007,100000.0,50000.0,Petrol,Individual,Manual,First Owner,1263,18,Low,Medium,0.0,Petrol_Manual,Hyundai,2.0,0
1263,CAR_001264,Volkswagen Passat 1.8 TSI MT,2010,480000.0,60000.0,Petrol,Individual,Manual,Second Owner,1264,15,Mid,High,0.0,Petrol_Manual,Volkswagen,6.0,0
1264,CAR_001265,Toyota Etios GD,2012,4461000.0,124000.0,Diesel,Individual,Manual,First Owner,1265,13,Low,High,0.0,Diesel_Manual,Toyota,2.42,1
1265,CAR_001266,Chevrolet Beat Diesel LS,2013,190000.0,60000.0,Petrol,Individual,Manual,First Owner,1266,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.71,1
1266,CAR_001267,Ford Figo Diesel LXI,2012,190000.0,130000.0,Diesel,Individual,Manual,Second Owner,1267,13,Low,High,0.0,Diesel_Manual,Ford,1.46,1
1267,CAR_001268,Ford EcoSport 1.5 Ti VCT MT Trend BSIV,2017,650000.0,60000.0,Petrol,Dealer,Manual,First Owner,1268,8,High,Low,0.0,Petrol_Manual,Ford,34.19,0
1268,CAR_001269,Toyota Corolla Executive (HE),2009,300000.0,23262.0,Petrol,Dealer,Manual,First Owner,1269,16,Low,Low,0.0,Petrol_Manual,Toyota,12.9,0
1269,CAR_001270,Ford Freestyle Titanium Diesel BSIV,2016,700000.0,29600.0,Diesel,Dealer,Manual,First Owner,1270,9,High,Low,0.0,Diesel_Manual,Ford,23.65,1
1270,CAR_001271,Hyundai i10 Sportz 1.1L,2015,310000.0,35925.0,Diesel,Dealer,Manual,First Owner,1271,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.63,0
1271,CAR_001272,Maruti Ignis 1.2 Zeta BSIV,2018,500000.0,12000.0,CNG,Individual,Manual,First Owner,1272,7,Mid,Low,0.0,Petrol_Manual,Maruti,41.67,0
1272,CAR_001273,Toyota Etios V,2013,415000.0,40771.0,Petrol,Dealer,Manual,First Owner,1273,12,Mid,Medium,0.0,Petrol_Manual,Toyota,10.18,0
1273,CAR_001274,Maruti Celerio VXI AT,2015,395000.0,30500.0,Petrol,Dealer,Automatic,Second Owner,1274,10,Mid,Medium,0.0,Petrol_Automatic,Maruti,12.95,0
1274,CAR_001275,Toyota Etios Liva 1.2 G,2014,430000.0,42000.0,Petrol,Dealer,Manual,Second Owner,1275,11,Low,Medium,0.0,Petrol_Manual,Toyota,10.24,0
1275,CAR_001276,Toyota Corolla H6,2007,215000.0,55800.0,Petrol,Dealer,Manual,Second Owner,1276,18,Low,Medium,0.0,Petrol_Manual,Toyota,3.85,0
1276,CAR_001277,Renault Duster RXL AWD,2014,550000.0,60000.0,Diesel,Dealer,Manual,First Owner,1277,11,Mid,Medium,0.0,Diesel_Manual,Renault,8.26,1
1277,CAR_001278,Volkswagen Polo 1.5 TDI Trendline,2014,415000.0,81358.0,Diesel,Dealer,Manual,First Owner,1278,11,Mid,High,0.0,Diesel_Manual,Volkswagen,5.1,1
1278,CAR_001279,Volkswagen Polo 1.5 TDI Comfortline,2015,450000.0,82695.0,Diesel,Dealer,Manual,First Owner,1279,10,Mid,High,0.0,Diesel_Manual,Volkswagen,5.44,1
1279,CAR_001280,Ford Figo Diesel EXI,2014,285000.0,68293.0,Diesel,Dealer,Manual,Second Owner,1280,11,Low,Medium,0.0,Diesel_Manual,Ford,4.17,1
1280,CAR_001281,Honda Jazz VX CVT,2016,580000.0,60000.0,Petrol,Dealer,Automatic,First Owner,1281,9,Mid,Medium,0.0,Petrol_Automatic,Honda,18.12,0
1281,CAR_001282,Maruti Swift Dzire Tour LDI,2014,4461000.0,190621.0,Diesel,Dealer,Manual,First Owner,1282,11,Low,Very High,0.0,Diesel_Manual,Maruti,1.57,1
1282,CAR_001283,Hyundai Verna CRDi 1.6 SX Option,2018,1200000.0,60000.0,Diesel,Individual,Manual,First Owner,1283,7,Premium,Medium,0.0,Diesel_Manual,Hyundai,20.0,1
1283,CAR_001284,Maruti 800 Std MPFi,2006,78000.0,64700.0,Petrol,Individual,Manual,First Owner,1284,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.21,0
1284,CAR_001285,Maruti 800 AC,2002,65000.0,100000.0,Petrol,Individual,Manual,Second Owner,1285,23,Low,High,0.0,Petrol_Manual,Maruti,0.65,0
1285,CAR_001286,Tata Zest Quadrajet 1.3 75PS XE,2016,400000.0,60000.0,Diesel,Individual,Manual,First Owner,1286,9,Mid,High,0.0,Diesel_Manual,Tata,4.52,1
1286,CAR_001287,Hyundai Verna 1.6 CRDI SX Option,2016,850000.0,75000.0,Diesel,Individual,Manual,First Owner,1287,9,High,High,0.0,Diesel_Manual,Hyundai,11.33,1
1287,CAR_001288,Volkswagen Vento Diesel Highline,2013,450000.0,60000.0,Diesel,Individual,Manual,Second Owner,1288,12,Mid,High,0.0,Diesel_Manual,Volkswagen,3.75,1
1288,CAR_001289,Mercedes-Benz E-Class E250 CDI Blue Efficiency,2012,4461000.0,35000.0,Diesel,Individual,Automatic,First Owner,1289,13,Premium,Medium,0.0,Diesel_Automatic,Mercedes-Benz,71.43,1
1289,CAR_001290,Toyota Etios 1.4 VXD,2017,780000.0,70000.0,Diesel,Individual,Manual,First Owner,1290,8,High,Medium,0.0,Diesel_Manual,Toyota,11.14,1
1290,CAR_001291,Toyota Fortuner 2.8 2WD AT BSIV,2017,3200000.0,57000.0,Diesel,Individual,Automatic,First Owner,1291,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,56.14,1
1291,CAR_001292,Maruti Alto 800 VXI,2020,350000.0,1000.0,Petrol,Individual,Manual,First Owner,1292,5,Mid,Medium,0.0,Petrol_Manual,Maruti,350.0,0
1292,CAR_001293,Mahindra Quanto C6,2013,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,1293,12,Low,High,0.0,Diesel_Manual,Mahindra,2.38,1
1293,CAR_001294,Volkswagen Vento 1.5 TDI Highline,2015,525000.0,60000.0,Diesel,Dealer,Manual,First Owner,1294,10,Low,Medium,0.0,Diesel_Manual,Volkswagen,5.36,1
1294,CAR_001295,Maruti Ciaz ZDi,2014,550000.0,90000.0,Diesel,Dealer,Manual,Second Owner,1295,11,Mid,High,0.0,Diesel_Manual,Maruti,6.11,1
1295,CAR_001296,Volkswagen Ameo 1.5 TDI Comfortline,2018,4461000.0,70000.0,Diesel,Individual,Manual,First Owner,1296,7,Mid,Medium,0.0,Diesel_Manual,Volkswagen,8.29,1
1296,CAR_001297,Tata Nano CX SE,2013,75000.0,25000.0,Petrol,Individual,Manual,Third Owner,1297,12,Low,Low,0.0,Petrol_Manual,Tata,3.0,0
1297,CAR_001298,Tata Manza Aura Quadrajet,2009,185000.0,60000.0,Diesel,Individual,Manual,Third Owner,1298,16,Low,Medium,0.0,Diesel_Manual,Tata,3.08,1
1298,CAR_001299,Maruti Omni E 8 Str STD,2007,110000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,1299,18,Low,High,0.0,Petrol_Manual,Maruti,1.22,0
1299,CAR_001300,Honda City 1.5 EXI,2004,180000.0,200000.0,Petrol,Individual,Manual,Second Owner,1300,21,Low,Very High,0.0,Petrol_Manual,Honda,0.9,0
1300,CAR_001301,Volkswagen Polo Diesel Comfortline 1.2L,2010,350000.0,110000.0,LPG,Individual,Manual,Second Owner,1301,15,Mid,High,0.0,Diesel_Manual,Volkswagen,3.18,1
1301,CAR_001302,Chevrolet Optra 1.6 LS,2005,60000.0,126000.0,Petrol,Individual,Manual,Second Owner,1302,20,Low,High,0.0,Petrol_Manual,Chevrolet,0.48,0
1302,CAR_001303,Mahindra TUV 300 T10,2018,695000.0,70000.0,Diesel,Individual,Manual,First Owner,1303,7,High,Medium,0.0,Diesel_Manual,Mahindra,9.93,1
1303,CAR_001304,Mahindra Alturas G4 4X2 AT BSIV,2019,2700000.0,5000.0,Diesel,Individual,Automatic,First Owner,1304,6,Premium,Low,0.0,Diesel_Automatic,Mahindra,540.0,1
1304,CAR_001305,Maruti Zen Estilo LX BSIII,2007,135000.0,74183.0,Petrol,Individual,Manual,Second Owner,1305,18,Low,High,0.0,Petrol_Manual,Maruti,1.82,0
1305,CAR_001306,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,640000.0,70000.0,Diesel,Individual,Manual,Second Owner,1306,10,Low,Medium,0.0,Diesel_Manual,Mahindra,9.14,1
1306,CAR_001307,Maruti Alto 800 VXI,2014,160000.0,90000.0,Petrol,Individual,Manual,First Owner,1307,11,Low,High,0.0,Petrol_Manual,Maruti,1.78,0
1307,CAR_001308,Tata Indica LXI,2004,80000.0,46000.0,Petrol,Individual,Manual,Second Owner,1308,21,Low,Medium,0.0,Petrol_Manual,Tata,1.74,0
1308,CAR_001309,Maruti Swift Dzire VDI,2013,340000.0,60000.0,Diesel,Individual,Manual,Second Owner,1309,12,Mid,High,0.0,Diesel_Manual,Maruti,3.09,1
1309,CAR_001310,Maruti Ritz VDi,2012,4461000.0,80000.0,Diesel,Individual,Manual,Second Owner,1310,13,Low,High,0.0,Diesel_Manual,Maruti,1.38,1
1310,CAR_001311,Tata Sumo Gold EX,2013,250000.0,90000.0,Diesel,Individual,Manual,Second Owner,1311,12,Low,High,0.0,Diesel_Manual,Tata,2.78,1
1311,CAR_001312,Toyota Innova Crysta 2.4 VX MT BSIV,2017,1300000.0,70000.0,Electric,Individual,Manual,First Owner,1312,8,Premium,Medium,0.0,Diesel_Manual,Toyota,18.57,1
1312,CAR_001313,Mahindra Quanto C6,2014,250000.0,1.0,Diesel,Individual,Manual,Second Owner,1313,11,Low,Low,0.0,Diesel_Manual,Mahindra,250000.0,1
1313,CAR_001314,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2009,450000.0,192000.0,Diesel,Individual,Manual,First Owner,1314,16,Mid,Very High,0.0,Diesel_Manual,Toyota,2.34,1
1314,CAR_001315,Maruti Wagon R LX Minor,2013,240000.0,70000.0,Petrol,Individual,Manual,Fourth & Above Owner,1315,12,Low,Medium,0.0,Petrol_Manual,Maruti,3.43,0
1315,CAR_001316,Mahindra Bolero DI DX 7 Seater,2007,225000.0,120000.0,Diesel,Individual,Manual,Second Owner,1316,18,Low,High,0.0,Diesel_Manual,Mahindra,1.88,1
1316,CAR_001317,Maruti Wagon R VXI Optional,2015,275000.0,60000.0,Petrol,Individual,Manual,Second Owner,1317,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.58,0
1317,CAR_001318,Tata Sumo Victa CX 7/9 Str BSII,2012,320000.0,120000.0,Diesel,Individual,Manual,Second Owner,1318,13,Mid,Medium,0.0,Diesel_Manual,Tata,2.67,1
1318,CAR_001319,Maruti Alto LX BSIII,2008,75000.0,120000.0,Petrol,Individual,Manual,First Owner,1319,17,Low,High,0.0,Petrol_Manual,Maruti,0.62,0
1319,CAR_001320,Tata Indica GLS BS IV,2007,85000.0,50000.0,Petrol,Individual,Manual,Second Owner,1320,18,Low,Medium,0.0,Petrol_Manual,Tata,1.7,0
1320,CAR_001321,Tata Indica Vista Aqua TDI BSIII,2009,90000.0,120000.0,Diesel,Individual,Manual,First Owner,1321,16,Low,High,0.0,Diesel_Manual,Tata,0.75,1
1321,CAR_001322,Maruti Wagon R LXI Minor,2007,4461000.0,80000.0,Petrol,Individual,Manual,Second Owner,1322,18,Low,High,0.0,Petrol_Manual,Maruti,1.06,0
1322,CAR_001323,Mahindra Scorpio M2DI,2008,4461000.0,80000.0,Diesel,Individual,Manual,Second Owner,1323,17,Mid,High,0.0,Diesel_Manual,Mahindra,4.38,1
1323,CAR_001324,Skoda Superb Elegance 2.0 TDI CR AT,2009,355000.0,100000.0,Diesel,Individual,Automatic,Second Owner,1324,16,Mid,High,0.0,Diesel_Automatic,Skoda,3.55,1
1324,CAR_001325,Maruti Swift VXI,2020,619000.0,1500.0,CNG,Individual,Manual,First Owner,1325,5,High,Low,0.0,Petrol_Manual,Maruti,412.67,0
1325,CAR_001326,Jeep Compass 2.0 Longitude Option BSIV,2018,1700000.0,50000.0,Diesel,Individual,Manual,First Owner,1326,7,Premium,Medium,0.0,Diesel_Manual,Jeep,34.0,1
1326,CAR_001327,Tata Indica DL,2006,55000.0,100000.0,Diesel,Individual,Manual,Second Owner,1327,19,Low,Medium,0.0,Diesel_Manual,Tata,0.55,1
1327,CAR_001328,Ford Fiesta 1.6 ZXi Leather,2006,95000.0,83411.0,Petrol,Individual,Manual,First Owner,1328,19,Low,High,0.0,Petrol_Manual,Ford,1.14,0
1328,CAR_001329,Volkswagen Vento Diesel Highline,2011,300000.0,120000.0,Diesel,Individual,Manual,Second Owner,1329,14,Low,High,0.0,Diesel_Manual,Volkswagen,2.5,1
1329,CAR_001330,Ford Fiesta 1.6 ZXi Leather,2006,81000.0,83411.0,Petrol,Individual,Manual,First Owner,1330,19,Low,High,0.0,Petrol_Manual,Ford,0.97,0
1330,CAR_001331,Skoda Superb Elegance 2.0 TDI CR AT,2015,1000000.0,80000.0,Diesel,Individual,Automatic,Second Owner,1331,10,High,High,0.0,Diesel_Automatic,Skoda,12.5,1
1331,CAR_001332,Maruti Ertiga VDI,2019,925000.0,50000.0,Diesel,Individual,Manual,First Owner,1332,6,Low,Medium,0.0,Diesel_Manual,Maruti,18.5,1
1332,CAR_001333,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,450000.0,40000.0,Diesel,Individual,Manual,First Owner,1333,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,11.25,1
1333,CAR_001334,Mahindra Scorpio EX,2014,600000.0,13270.0,LPG,Individual,Manual,First Owner,1334,11,Mid,Low,0.0,Diesel_Manual,Mahindra,45.21,1
1334,CAR_001335,Hyundai Santro Xing XL AT eRLX Euro III,2006,85000.0,100000.0,Petrol,Individual,Automatic,Third Owner,1335,19,Low,High,0.0,Petrol_Automatic,Hyundai,0.85,0
1335,CAR_001336,Mahindra Thar DI 4X4 PS,2016,700000.0,50000.0,Diesel,Individual,Manual,First Owner,1336,9,High,Medium,0.0,Diesel_Manual,Mahindra,14.0,1
1336,CAR_001337,Hyundai Grand i10 1.2 Kappa Magna BSIV,2017,486000.0,60000.0,Petrol,Individual,Manual,First Owner,1337,8,Mid,Low,0.0,Petrol_Manual,Hyundai,18.34,0
1337,CAR_001338,Maruti Swift VDI,2013,400000.0,90000.0,Diesel,Individual,Manual,First Owner,1338,12,Mid,High,0.0,Diesel_Manual,Maruti,4.44,1
1338,CAR_001339,Ford Fiesta 1.4 Duratec ZXI,2008,120000.0,100000.0,Petrol,Individual,Manual,Third Owner,1339,17,Low,High,0.0,Petrol_Manual,Ford,1.2,0
1339,CAR_001340,Maruti Alto 800 LXI,2018,275000.0,35000.0,Petrol,Individual,Manual,First Owner,1340,7,Low,Medium,0.0,Petrol_Manual,Maruti,7.86,0
1340,CAR_001341,Hyundai i20 Asta Option 1.4 CRDi,2016,580000.0,57000.0,Diesel,Individual,Manual,First Owner,1341,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.18,1
1341,CAR_001342,Maruti Swift Dzire LXI,2015,375000.0,70000.0,Petrol,Individual,Manual,First Owner,1342,10,Mid,Medium,0.0,Petrol_Manual,Maruti,5.36,0
1342,CAR_001343,Maruti Wagon R VXI BS IV with ABS,2015,229999.0,100000.0,Petrol,Individual,Manual,Third Owner,1343,10,Low,High,0.0,Petrol_Manual,Maruti,2.3,0
1343,CAR_001344,Maruti Alto LX,2010,110000.0,88000.0,Petrol,Individual,Manual,Second Owner,1344,15,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
1344,CAR_001345,Maruti Swift Dzire VDI,2017,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,1345,8,Mid,Medium,0.0,Diesel_Manual,Maruti,10.0,1
1345,CAR_001346,Maruti A-Star Vxi,2009,4461000.0,76000.0,Petrol,Individual,Manual,Second Owner,1346,16,Low,High,0.0,Petrol_Manual,Maruti,1.58,0
1346,CAR_001347,Ford Figo Diesel Titanium,2010,200000.0,100000.0,Diesel,Individual,Manual,Second Owner,1347,15,Low,High,0.0,Diesel_Manual,Ford,2.0,1
1347,CAR_001348,Hyundai Verna 1.6 SX CRDi (O),2012,450000.0,110000.0,Diesel,Individual,Manual,Second Owner,1348,13,Mid,High,0.0,Diesel_Manual,Hyundai,4.09,1
1348,CAR_001349,Hyundai Verna 1.6 SX CRDi (O),2012,550000.0,80000.0,Diesel,Individual,Manual,Second Owner,1349,13,Mid,High,0.0,Diesel_Manual,Hyundai,6.88,1
1349,CAR_001350,Maruti Alto LX,2012,220000.0,80000.0,Petrol,Individual,Manual,Second Owner,1350,13,Low,High,0.0,Petrol_Manual,Maruti,2.75,0
1350,CAR_001351,Maruti 800 EX,2001,40000.0,30000.0,Petrol,Individual,Manual,First Owner,1351,24,Low,Low,0.0,Petrol_Manual,Maruti,1.33,0
1351,CAR_001352,Renault KWID Climber 1.0 AMT,2017,300000.0,20000.0,Electric,Individual,Automatic,First Owner,1352,8,Low,Low,0.0,Petrol_Automatic,Renault,15.0,0
1352,CAR_001353,Mahindra Bolero Power Plus SLE,2018,630000.0,40000.0,Diesel,Individual,Manual,First Owner,1353,7,High,Medium,0.0,Diesel_Manual,Mahindra,15.75,1
1353,CAR_001354,Maruti Swift Dzire VDI,2017,610000.0,40000.0,Diesel,Individual,Manual,Second Owner,1354,8,High,Medium,0.0,Diesel_Manual,Maruti,15.25,1
1354,CAR_001355,Hyundai Xcent 1.2 VTVT S,2019,660000.0,7000.0,Petrol,Individual,Manual,Second Owner,1355,6,Low,Low,0.0,Petrol_Manual,Hyundai,94.29,0
1355,CAR_001356,Maruti Wagon R LXI LPG BSIV,2012,220000.0,80000.0,Petrol,Individual,Manual,Second Owner,1356,13,Low,High,0.0,LPG_Manual,Maruti,2.75,0
1356,CAR_001357,Nissan Micra Diesel XV Premium,2011,138000.0,90000.0,Diesel,Individual,Manual,Second Owner,1357,14,Low,High,0.0,Diesel_Manual,Nissan,1.53,1
1357,CAR_001358,Nissan Micra Diesel XV Premium,2013,210000.0,13770.0,Diesel,Individual,Manual,First Owner,1358,12,Low,Low,0.0,Diesel_Manual,Nissan,15.25,1
1358,CAR_001359,Maruti Zen LXi - BS III,2005,125000.0,34000.0,Petrol,Dealer,Manual,First Owner,1359,20,Low,Medium,0.0,Petrol_Manual,Maruti,3.68,0
1359,CAR_001360,Maruti Alto LXi BSIII,2008,125000.0,102000.0,Petrol,Dealer,Manual,First Owner,1360,17,Low,High,0.0,Petrol_Manual,Maruti,1.23,0
1360,CAR_001361,Honda City i DTEC SV,2014,725000.0,110000.0,Diesel,Individual,Manual,First Owner,1361,11,High,High,0.0,Diesel_Manual,Honda,6.59,1
1361,CAR_001362,Maruti Alto K10 2010-2014 VXI,2011,170000.0,90000.0,Petrol,Individual,Manual,Second Owner,1362,14,Low,High,0.0,Petrol_Manual,Maruti,1.89,0
1362,CAR_001363,Toyota Innova 2.5 VX (Diesel) 7 Seater,2015,1200000.0,70000.0,Diesel,Individual,Manual,Third Owner,1363,10,Premium,Medium,0.0,Diesel_Manual,Toyota,17.14,1
1363,CAR_001364,Maruti Wagon R VXI BSIII,2003,70000.0,90000.0,Petrol,Individual,Manual,Second Owner,1364,22,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
1364,CAR_001365,Maruti Alto LXi,2007,100000.0,70000.0,Petrol,Individual,Manual,Second Owner,1365,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.43,0
1365,CAR_001366,Honda Civic 1.8 (E) MT,2006,300000.0,80000.0,Petrol,Individual,Manual,Second Owner,1366,19,Low,High,0.0,Petrol_Manual,Honda,3.75,0
1366,CAR_001367,Maruti Swift VDI,2013,450000.0,100000.0,Diesel,Individual,Manual,Second Owner,1367,12,Mid,High,0.0,Diesel_Manual,Maruti,4.5,1
1367,CAR_001368,Maruti Swift Dzire VDI,2015,530000.0,100000.0,Diesel,Individual,Manual,First Owner,1368,10,Mid,High,0.0,Diesel_Manual,Maruti,5.3,1
1368,CAR_001369,Maruti Zen Estilo 1.1 LXI BSIII,2007,100000.0,143000.0,Petrol,Individual,Manual,Third Owner,1369,18,Low,High,0.0,Petrol_Manual,Maruti,0.7,0
1369,CAR_001370,Honda City 1.5 V MT,2010,350000.0,115000.0,Petrol,Individual,Manual,Second Owner,1370,15,Mid,High,0.0,Petrol_Manual,Honda,3.04,0
1370,CAR_001371,Tata Manza ELAN Quadrajet BS IV,2011,165000.0,100000.0,Diesel,Individual,Manual,Second Owner,1371,14,Low,High,0.0,Diesel_Manual,Tata,1.65,1
1371,CAR_001372,Maruti Swift Dzire ZDI,2019,475000.0,120000.0,Diesel,Individual,Manual,Second Owner,1372,6,Mid,High,0.0,Diesel_Manual,Maruti,3.96,1
1372,CAR_001373,Mahindra Xylo E4 ABS BS IV,2013,350000.0,42000.0,Diesel,Individual,Manual,First Owner,1373,12,Low,Medium,0.0,Diesel_Manual,Mahindra,8.33,1
1373,CAR_001374,Renault KWID RXL,2016,229999.0,60000.0,CNG,Individual,Manual,First Owner,1374,9,Low,Medium,0.0,Petrol_Manual,Renault,6.05,0
1374,CAR_001375,Honda WR-V i-DTEC VX,2018,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,1375,7,High,Medium,0.0,Diesel_Manual,Honda,22.86,1
1375,CAR_001376,Maruti Swift Dzire ZDI,2014,475000.0,120000.0,Diesel,Individual,Manual,Second Owner,1376,11,Mid,Medium,0.0,Diesel_Manual,Maruti,3.96,1
1376,CAR_001377,Nissan Terrano XL 110 PS,2014,425000.0,60000.0,Diesel,Individual,Manual,First Owner,1377,11,Mid,Medium,0.0,Diesel_Manual,Nissan,3.1,1
1377,CAR_001378,Maruti Ciaz VDi Plus,2015,660000.0,97000.0,Diesel,Individual,Manual,Second Owner,1378,10,High,High,0.0,Diesel_Manual,Maruti,6.8,1
1378,CAR_001379,Honda Amaze S CVT Petrol,2019,4461000.0,10000.0,Petrol,Individual,Automatic,First Owner,1379,6,High,Low,0.0,Petrol_Automatic,Honda,61.0,0
1379,CAR_001380,Honda Brio S MT,2013,240000.0,50000.0,Petrol,Individual,Manual,Third Owner,1380,12,Low,Medium,0.0,Petrol_Manual,Honda,4.8,0
1380,CAR_001381,Maruti Esteem Lxi,2005,70000.0,60000.0,LPG,Individual,Manual,Second Owner,1381,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.17,0
1381,CAR_001382,Toyota Corolla Altis Diesel D4DGL,2011,550000.0,60000.0,Diesel,Individual,Manual,First Owner,1382,14,Mid,High,0.0,Diesel_Manual,Toyota,4.58,1
1382,CAR_001383,Tata Indica Vista Aura 1.2 Safire,2010,100000.0,60000.0,Petrol,Individual,Manual,Second Owner,1383,15,Low,High,0.0,Petrol_Manual,Tata,0.83,0
1383,CAR_001384,Mahindra Bolero SLE,2010,350000.0,90000.0,Diesel,Individual,Manual,Second Owner,1384,15,Mid,High,0.0,Diesel_Manual,Mahindra,3.89,1
1384,CAR_001385,Hyundai EON Era Plus,2013,200000.0,120000.0,Petrol,Individual,Manual,First Owner,1385,12,Low,Medium,0.0,Petrol_Manual,Hyundai,1.67,0
1385,CAR_001386,Hyundai Elite i20 Asta Option CVT BSIV,2019,720000.0,3000.0,Petrol,Individual,Automatic,First Owner,1386,6,High,Low,0.0,Petrol_Automatic,Hyundai,240.0,0
1386,CAR_001387,Maruti Ritz VDi,2014,350000.0,133000.0,Diesel,Individual,Manual,First Owner,1387,11,Mid,High,0.0,Diesel_Manual,Maruti,2.63,1
1387,CAR_001388,Maruti Wagon R VXI BS IV,2016,400000.0,25000.0,Petrol,Individual,Manual,First Owner,1388,9,Mid,Low,0.0,Petrol_Manual,Maruti,16.0,0
1388,CAR_001389,Hyundai i20 Active 1.2 S,2017,570000.0,35000.0,Petrol,Individual,Manual,First Owner,1389,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,16.29,0
1389,CAR_001390,Maruti Omni MPI STD BSIV,2015,130000.0,70000.0,Petrol,Individual,Manual,First Owner,1390,10,Low,Medium,0.0,Petrol_Manual,Maruti,1.86,0
1390,CAR_001391,Maruti Alto 800 LXI,2016,210000.0,52000.0,Electric,Individual,Manual,First Owner,1391,9,Low,Medium,0.0,Petrol_Manual,Maruti,4.04,0
1391,CAR_001392,Hyundai Creta 1.4 E Plus,2018,800000.0,52000.0,Diesel,Individual,Manual,First Owner,1392,7,High,Medium,0.0,Diesel_Manual,Hyundai,15.38,1
1392,CAR_001393,Honda Jazz 1.5 E i DTEC,2017,700000.0,39000.0,Petrol,Individual,Manual,First Owner,1393,8,High,Medium,0.0,Diesel_Manual,Honda,17.95,1
1393,CAR_001394,Tata Indica Vista Quadrajet VX,2011,200000.0,28689.0,Diesel,Individual,Manual,First Owner,1394,14,Low,Low,0.0,Diesel_Manual,Tata,6.97,1
1394,CAR_001395,Tata Aria Pure LX 4x2,2014,495000.0,90000.0,Diesel,Individual,Manual,Second Owner,1395,11,Mid,Medium,0.0,Diesel_Manual,Tata,5.5,1
1395,CAR_001396,Ford Fiesta Classic 1.4 Duratorq CLXI,2014,325000.0,60000.0,Diesel,Dealer,Manual,First Owner,1396,11,Mid,High,0.0,Diesel_Manual,Ford,4.05,1
1396,CAR_001397,Ford Freestyle Trend Petrol BSIV,2019,650000.0,8000.0,Petrol,Individual,Manual,First Owner,1397,6,High,Low,0.0,Petrol_Manual,Ford,81.25,0
1397,CAR_001398,Mahindra Jeep MM 775 XDB,2000,4461000.0,50000.0,Diesel,Individual,Manual,Second Owner,1398,25,Low,Medium,0.0,Diesel_Manual,Mahindra,3.1,1
1398,CAR_001399,Maruti 800 AC BSIII,2007,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,1399,18,Low,High,0.0,Petrol_Manual,Maruti,0.83,0
1399,CAR_001400,Hyundai Grand i10 1.2 CRDi Magna,2015,350000.0,61658.0,Diesel,Dealer,Manual,First Owner,1400,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,5.68,1
1400,CAR_001401,Maruti Zen Estilo LX BSIV,2010,90000.0,60000.0,Petrol,Individual,Manual,Third Owner,1401,15,Low,Medium,0.0,Petrol_Manual,Maruti,1.8,0
1401,CAR_001402,Maruti Ertiga SHVS VDI,2016,600000.0,80000.0,Diesel,Individual,Manual,First Owner,1402,9,Mid,High,0.0,Diesel_Manual,Maruti,7.5,1
1402,CAR_001403,Mahindra Scorpio SLE BSIII,2010,4461000.0,185000.0,Diesel,Individual,Manual,First Owner,1403,15,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.16,1
1403,CAR_001404,Maruti Swift VDI BSIV,2014,370000.0,57000.0,Diesel,Individual,Manual,Second Owner,1404,11,Mid,Medium,0.0,Diesel_Manual,Maruti,6.49,1
1404,CAR_001405,Maruti 800 DX,2000,50000.0,60000.0,Petrol,Individual,Manual,First Owner,1405,25,Low,Medium,0.0,Petrol_Manual,Maruti,0.83,0
1405,CAR_001406,Honda City 1.5 GXI,2004,125000.0,120000.0,Petrol,Individual,Manual,Second Owner,1406,21,Low,High,0.0,Petrol_Manual,Honda,1.04,0
1406,CAR_001407,Honda CR-V Diesel 4WD,2018,4461000.0,26000.0,Diesel,Individual,Automatic,First Owner,1407,7,Low,Low,0.0,Diesel_Automatic,Honda,69.23,1
1407,CAR_001408,Hyundai Grand i10 1.2 Kappa Magna AT,2019,650000.0,20000.0,Petrol,Individual,Automatic,First Owner,1408,6,High,Low,0.0,Petrol_Automatic,Hyundai,32.5,0
1408,CAR_001409,Maruti Ertiga BSIV VXI AT,2017,800000.0,60000.0,Petrol,Individual,Automatic,First Owner,1409,8,High,Medium,0.0,Petrol_Automatic,Maruti,26.14,0
1409,CAR_001410,Volkswagen Polo 1.0 TSI Highline Plus,2020,802000.0,5000.0,Petrol,Individual,Manual,First Owner,1410,5,High,Low,0.0,Petrol_Manual,Volkswagen,160.4,0
1410,CAR_001411,Audi A4 35 TDI Premium,2015,2300000.0,35000.0,Diesel,Individual,Automatic,Third Owner,1411,10,Low,Medium,0.0,Diesel_Automatic,Audi,65.71,1
1411,CAR_001412,Hyundai Xcent 1.1 CRDi SX,2016,430000.0,35000.0,CNG,Individual,Manual,First Owner,1412,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,12.29,1
1412,CAR_001413,Hyundai Verna 1.6 SX VTVT,2017,650000.0,23000.0,Petrol,Individual,Manual,First Owner,1413,8,Low,Low,0.0,Petrol_Manual,Hyundai,28.26,0
1413,CAR_001414,Mahindra Scorpio 2.6 SLX Turbo 7 Seater,2004,170000.0,120000.0,Diesel,Individual,Manual,First Owner,1414,21,Low,High,0.0,Diesel_Manual,Mahindra,1.42,1
1414,CAR_001415,Skoda Superb Elegance 2.0 TDI CR AT,2011,450000.0,235000.0,Diesel,Individual,Automatic,First Owner,1415,14,Mid,Very High,0.0,Diesel_Automatic,Skoda,1.91,1
1415,CAR_001416,Maruti Alto 800 LXI,2013,250000.0,87000.0,Petrol,Individual,Manual,First Owner,1416,12,Low,High,0.0,Petrol_Manual,Maruti,2.87,0
1416,CAR_001417,Ford Ecosport 1.0 Ecoboost Titanium,2014,550000.0,72000.0,Petrol,Individual,Manual,Second Owner,1417,11,Low,Medium,0.0,Petrol_Manual,Ford,7.64,0
1417,CAR_001418,Jaguar XF 3.0 Litre S Premium Luxury,2013,2000000.0,60000.0,Diesel,Individual,Automatic,Second Owner,1418,12,Premium,Medium,0.0,Diesel_Automatic,Jaguar,31.25,1
1418,CAR_001419,Maruti Zen Estilo VXI BSIII,2007,140000.0,100000.0,Petrol,Individual,Manual,Second Owner,1419,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.4,0
1419,CAR_001420,Chevrolet Beat LT,2012,155000.0,12000.0,Petrol,Individual,Manual,First Owner,1420,13,Low,Low,0.0,Petrol_Manual,Chevrolet,12.92,0
1420,CAR_001421,Hyundai Accent Executive,2010,120000.0,67000.0,Petrol,Individual,Manual,Third Owner,1421,15,Low,Medium,0.0,Petrol_Manual,Hyundai,1.79,0
1421,CAR_001422,Toyota Innova 2.5 E 8 STR,2012,287000.0,70000.0,Diesel,Individual,Manual,First Owner,1422,13,Low,Medium,0.0,Diesel_Manual,Toyota,4.1,1
1422,CAR_001423,Maruti Baleno Alpha 1.3,2016,675000.0,60000.0,Diesel,Individual,Manual,First Owner,1423,9,High,Medium,0.0,Diesel_Manual,Maruti,11.25,1
1423,CAR_001424,Hyundai i10 Magna 1.2 iTech SE,2011,235000.0,74500.0,Petrol,Individual,Manual,Second Owner,1424,14,Low,High,0.0,Petrol_Manual,Hyundai,3.15,0
1424,CAR_001425,Hyundai i20 1.4 CRDi Asta,2011,4461000.0,118700.0,Diesel,Individual,Manual,Second Owner,1425,14,Low,High,0.0,Diesel_Manual,Hyundai,2.53,1
1425,CAR_001426,Nissan Micra XL CVT,2015,434999.0,40000.0,Petrol,Individual,Automatic,Second Owner,1426,10,Mid,Medium,0.0,Petrol_Automatic,Nissan,10.87,0
1426,CAR_001427,Mahindra Scorpio VLX AT 2WD BSIII,2004,225000.0,60000.0,Diesel,Individual,Automatic,Third Owner,1427,21,Low,Very High,0.0,Diesel_Automatic,Mahindra,1.01,1
1427,CAR_001428,Toyota Corolla AE,2006,160000.0,140000.0,Petrol,Individual,Manual,Second Owner,1428,19,Low,High,0.0,Petrol_Manual,Toyota,1.14,0
1428,CAR_001429,Hyundai Grand i10 Nios Sportz,2020,600000.0,5000.0,Petrol,Individual,Manual,First Owner,1429,5,Mid,Low,0.0,Petrol_Manual,Hyundai,120.0,0
1429,CAR_001430,Chevrolet Tavera Neo 3 LT 9 Seats BSIII,2016,4461000.0,100000.0,LPG,Individual,Manual,Second Owner,1430,9,Mid,High,0.0,Diesel_Manual,Chevrolet,4.7,1
1430,CAR_001431,Maruti Swift Dzire VXi,2011,300000.0,100000.0,Petrol,Individual,Manual,Second Owner,1431,14,Low,High,0.0,Petrol_Manual,Maruti,3.0,0
1431,CAR_001432,Maruti Baleno Delta 1.2,2018,600000.0,60000.0,Electric,Individual,Manual,First Owner,1432,7,Mid,Low,0.0,Petrol_Manual,Maruti,300.0,0
1432,CAR_001433,Hyundai Grand i10 Nios AMT Magna,2020,4461000.0,60000.0,Petrol,Individual,Automatic,First Owner,1433,5,High,Low,0.0,Petrol_Automatic,Hyundai,160.0,0
1433,CAR_001434,Tata Indica DLS,2006,50000.0,120000.0,Diesel,Individual,Manual,Fourth & Above Owner,1434,19,Low,High,0.0,Diesel_Manual,Tata,0.42,1
1434,CAR_001435,Hyundai Verna 1.6 VTVT S,2015,4461000.0,35000.0,Petrol,Individual,Manual,First Owner,1435,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,17.14,0
1435,CAR_001436,Hyundai i10 Magna LPG,2014,290000.0,29000.0,LPG,Individual,Manual,First Owner,1436,11,Low,Low,0.0,LPG_Manual,Hyundai,10.0,0
1436,CAR_001437,Hyundai Verna 1.6 VTVT S,2015,600000.0,30000.0,Petrol,Individual,Manual,First Owner,1437,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,20.0,0
1437,CAR_001438,Tata Manza Aqua Quadrajet BS IV,2012,190000.0,120000.0,Diesel,Individual,Manual,First Owner,1438,13,Low,High,0.0,Diesel_Manual,Tata,1.58,1
1438,CAR_001439,Hyundai i10 Era 1.1,2008,160000.0,60000.0,Petrol,Individual,Manual,Third Owner,1439,17,Low,High,0.0,Petrol_Manual,Hyundai,1.45,0
1439,CAR_001440,Maruti Alto K10 VXI,2017,325000.0,60000.0,Petrol,Individual,Manual,First Owner,1440,8,Mid,Low,0.0,Petrol_Manual,Maruti,21.67,0
1440,CAR_001441,Mahindra Bolero DI,2006,200000.0,73756.0,Diesel,Individual,Manual,First Owner,1441,19,Low,High,0.0,Diesel_Manual,Mahindra,2.71,1
1441,CAR_001442,Toyota Fortuner 2.8 2WD AT BSIV,2018,2800000.0,40000.0,Diesel,Individual,Automatic,First Owner,1442,7,Premium,Medium,0.0,Diesel_Automatic,Toyota,70.0,1
1442,CAR_001443,Chevrolet Spark 1.0 LS,2011,125000.0,60000.0,Petrol,Individual,Manual,Third Owner,1443,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.08,0
1443,CAR_001444,Hyundai i10 Magna 1.1,2008,155000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,1444,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.58,0
1444,CAR_001445,Hyundai i20 Asta 1.4 CRDi,2015,4461000.0,90000.0,Diesel,Individual,Manual,First Owner,1445,10,Mid,High,0.0,Diesel_Manual,Hyundai,6.11,1
1445,CAR_001446,Maruti Wagon R LX BSIII,2009,130000.0,70000.0,Petrol,Individual,Manual,Fourth & Above Owner,1446,16,Low,Medium,0.0,Petrol_Manual,Maruti,1.86,0
1446,CAR_001447,Maruti 800 AC,2002,80000.0,70000.0,Petrol,Individual,Manual,Second Owner,1447,23,Low,Medium,0.0,Petrol_Manual,Maruti,1.14,0
1447,CAR_001448,Toyota Etios Cross 1.4L GD,2015,500000.0,80000.0,Diesel,Individual,Manual,Second Owner,1448,10,Mid,High,0.0,Diesel_Manual,Toyota,6.25,1
1448,CAR_001449,Renault KWID RXT BSIV,2019,4461000.0,27000.0,Petrol,Individual,Manual,First Owner,1449,6,Low,Low,0.0,Petrol_Manual,Renault,9.26,0
1449,CAR_001450,Mahindra XUV500 W8 2WD,2012,580000.0,70000.0,Diesel,Individual,Manual,First Owner,1450,13,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.29,1
1450,CAR_001451,Daewoo Matiz SD,2002,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,1451,23,Low,Medium,0.0,Petrol_Manual,Daewoo,1.2,0
1451,CAR_001452,Toyota Etios GD,2012,300000.0,185000.0,Diesel,Individual,Manual,First Owner,1452,13,Low,Very High,0.0,Diesel_Manual,Toyota,1.62,1
1452,CAR_001453,Ford Figo Diesel Titanium,2014,200000.0,60000.0,Diesel,Individual,Manual,Second Owner,1453,11,Low,High,0.0,Diesel_Manual,Ford,2.22,1
1453,CAR_001454,Tata Hexa XT,2017,1200000.0,60000.0,Diesel,Individual,Manual,First Owner,1454,8,Premium,Low,0.0,Diesel_Manual,Tata,73.17,1
1454,CAR_001455,Toyota Etios Cross 1.4L GD,2015,350000.0,80000.0,Diesel,Individual,Manual,Second Owner,1455,10,Mid,High,0.0,Diesel_Manual,Toyota,4.38,1
1455,CAR_001456,Mahindra XUV500 W8 4WD,2012,600000.0,80000.0,Diesel,Individual,Manual,Second Owner,1456,13,Mid,High,0.0,Diesel_Manual,Mahindra,7.5,1
1456,CAR_001457,Honda Jazz V,2019,4461000.0,4000.0,Petrol,Individual,Manual,First Owner,1457,6,High,Low,0.0,Petrol_Manual,Honda,162.5,0
1457,CAR_001458,Maruti A-Star Lxi,2008,120000.0,60000.0,Petrol,Individual,Manual,First Owner,1458,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
1458,CAR_001459,Tata New Safari DICOR 2.2 EX 4x2,2010,180000.0,120000.0,Diesel,Individual,Manual,Second Owner,1459,15,Low,High,0.0,Diesel_Manual,Tata,1.5,1
1459,CAR_001460,Hyundai Verna VTVT 1.6 AT SX Option,2018,950000.0,41395.0,Petrol,Individual,Automatic,First Owner,1460,7,High,Medium,0.0,Petrol_Automatic,Hyundai,22.95,0
1460,CAR_001461,Tata Manza Aura Quadrajet BS IV,2012,250999.0,35000.0,Diesel,Individual,Manual,Second Owner,1461,13,Low,Medium,0.0,Diesel_Manual,Tata,7.17,1
1461,CAR_001462,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,1462,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
1462,CAR_001463,Audi A8 L 3.0 TDI quattro,2009,1250000.0,47000.0,Diesel,Dealer,Automatic,Second Owner,1463,16,Premium,Medium,0.0,Diesel_Automatic,Audi,26.6,1
1463,CAR_001464,Chevrolet Beat LT,2013,165000.0,50000.0,Petrol,Individual,Manual,First Owner,1464,12,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.3,0
1464,CAR_001465,Hyundai Elantra SX,2015,700000.0,48000.0,Petrol,Dealer,Manual,First Owner,1465,10,High,Medium,0.0,Petrol_Manual,Hyundai,14.58,0
1465,CAR_001466,Mahindra XUV500 W8 4WD,2015,750000.0,53000.0,Diesel,Dealer,Manual,First Owner,1466,10,High,Medium,0.0,Diesel_Manual,Mahindra,14.15,1
1466,CAR_001467,Mahindra Renault Logan 1.5 DLS,2008,89999.0,213000.0,Diesel,Individual,Manual,First Owner,1467,17,Low,Very High,0.0,Diesel_Manual,Mahindra,0.42,1
1467,CAR_001468,Datsun redi-GO AMT 1.0 T Option,2018,4461000.0,15000.0,Petrol,Dealer,Automatic,First Owner,1468,7,Low,Low,0.0,Petrol_Automatic,Datsun,19.33,0
1468,CAR_001469,Tata Nano Cx BSIII,2014,45000.0,7000.0,Petrol,Individual,Manual,Second Owner,1469,11,Low,Low,0.0,Petrol_Manual,Tata,6.43,0
1469,CAR_001470,Skoda Superb Ambition 2.0 TDI CR AT,2013,675000.0,88000.0,Diesel,Dealer,Automatic,First Owner,1470,12,High,High,0.0,Diesel_Automatic,Skoda,7.67,1
1470,CAR_001471,Nissan Terrano XL Plus 85 PS,2015,530000.0,55000.0,Diesel,Dealer,Manual,First Owner,1471,10,Mid,Medium,0.0,Diesel_Manual,Nissan,9.64,1
1471,CAR_001472,BMW X1 sDrive20d M Sport,2014,1485000.0,48000.0,Diesel,Dealer,Automatic,First Owner,1472,11,Premium,Medium,0.0,Diesel_Automatic,BMW,30.94,1
1472,CAR_001473,Chevrolet Enjoy TCDi LT 7 Seater,2014,325000.0,71014.0,Diesel,Dealer,Manual,First Owner,1473,11,Mid,High,0.0,Diesel_Manual,Chevrolet,4.58,1
1473,CAR_001474,Honda Amaze S Diesel,2015,400000.0,60000.0,Diesel,Dealer,Manual,First Owner,1474,10,Mid,Medium,0.0,Diesel_Manual,Honda,6.67,1
1474,CAR_001475,Mahindra KUV 100 D75 K6 Plus,2017,380000.0,71000.0,CNG,Dealer,Manual,First Owner,1475,8,Mid,High,0.0,Diesel_Manual,Mahindra,5.35,1
1475,CAR_001476,Honda City i-VTEC CVT VX,2016,750000.0,45000.0,Petrol,Dealer,Automatic,First Owner,1476,9,High,Medium,0.0,Petrol_Automatic,Honda,16.67,0
1476,CAR_001477,Honda Brio VX,2017,4461000.0,60000.0,Petrol,Dealer,Manual,First Owner,1477,8,Low,Medium,0.0,Petrol_Manual,Honda,8.67,0
1477,CAR_001478,Maruti Alto LXi,2009,110000.0,20000.0,Petrol,Individual,Manual,First Owner,1478,16,Low,Low,0.0,Petrol_Manual,Maruti,5.5,0
1478,CAR_001479,MG Hector Sharp Diesel MT BSIV,2019,1825000.0,14000.0,Diesel,Dealer,Manual,First Owner,1479,6,Premium,Medium,0.0,Diesel_Manual,MG,130.36,1
1479,CAR_001480,Audi A4 35 TDI Premium Plus,2019,3256000.0,17000.0,Diesel,Dealer,Automatic,First Owner,1480,6,Premium,Low,0.0,Diesel_Automatic,Audi,191.53,1
1480,CAR_001481,Renault Scala Diesel RxL,2015,370000.0,60000.0,Diesel,Dealer,Manual,First Owner,1481,10,Low,Medium,0.0,Diesel_Manual,Renault,6.17,1
1481,CAR_001482,Chevrolet Beat Diesel LT,2013,210000.0,80000.0,Diesel,Individual,Manual,Second Owner,1482,12,Low,High,0.0,Diesel_Manual,Chevrolet,2.62,1
1482,CAR_001483,Skoda Laura Ambiente 1.9 PD,2007,4461000.0,181000.0,Diesel,Individual,Manual,Fourth & Above Owner,1483,18,Low,Very High,0.0,Diesel_Manual,Skoda,0.97,1
1483,CAR_001484,Hyundai Verna 1.6 SX CRDi (O),2012,575000.0,89550.0,Diesel,Individual,Manual,Second Owner,1484,13,Mid,Medium,0.0,Diesel_Manual,Hyundai,6.42,1
1484,CAR_001485,Mahindra Xylo D4,2010,320000.0,90000.0,Diesel,Individual,Manual,Third Owner,1485,15,Mid,High,0.0,Diesel_Manual,Mahindra,3.56,1
1485,CAR_001486,Maruti Swift Ldi BSIV,2010,280000.0,149500.0,Diesel,Individual,Manual,Third Owner,1486,15,Low,High,0.0,Diesel_Manual,Maruti,1.87,1
1486,CAR_001487,Renault Duster 85PS Diesel RxL Optional,2013,500000.0,60000.0,Diesel,Individual,Manual,Second Owner,1487,12,Mid,Very High,0.0,Diesel_Manual,Renault,2.63,1
1487,CAR_001488,Ford Fiesta Diesel Style,2013,380000.0,83000.0,Diesel,Dealer,Manual,First Owner,1488,12,Mid,High,0.0,Diesel_Manual,Ford,4.58,1
1488,CAR_001489,Maruti 800 AC BSIII,2005,120000.0,20000.0,Petrol,Individual,Manual,Second Owner,1489,20,Low,Low,0.0,Petrol_Manual,Maruti,6.0,0
1489,CAR_001490,Maruti Eeco 5 Seater AC BSIV,2018,395000.0,20000.0,Petrol,Individual,Manual,First Owner,1490,7,Mid,Low,0.0,Petrol_Manual,Maruti,19.75,0
1490,CAR_001491,Skoda Laura 1.9 TDI MT Ambiente,2010,360000.0,160000.0,Diesel,Individual,Manual,First Owner,1491,15,Mid,Very High,0.0,Diesel_Manual,Skoda,2.25,1
1491,CAR_001492,Nissan Terrano XL Plus 85 PS,2015,451000.0,100000.0,LPG,Individual,Manual,First Owner,1492,10,Mid,High,0.0,Diesel_Manual,Nissan,4.51,1
1492,CAR_001493,Chevrolet Spark 1.0 LS,2010,175000.0,45000.0,Petrol,Individual,Manual,Second Owner,1493,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.89,0
1493,CAR_001494,Chevrolet Spark 1.0 LS,2009,170000.0,44800.0,Petrol,Individual,Manual,Second Owner,1494,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.79,0
1494,CAR_001495,Hyundai Santro Xing XL eRLX Euro III,2006,100000.0,40000.0,Petrol,Individual,Manual,Second Owner,1495,19,Low,Medium,0.0,Petrol_Manual,Hyundai,2.5,0
1495,CAR_001496,Maruti Omni MPI STD BSIV,2014,150000.0,120000.0,Petrol,Individual,Manual,Third Owner,1496,11,Low,Medium,0.0,Petrol_Manual,Maruti,1.25,0
1496,CAR_001497,Tata Indica Vista Quadrajet LX,2011,149000.0,10000.0,Electric,Individual,Manual,Second Owner,1497,14,Low,Low,0.0,Diesel_Manual,Tata,14.9,1
1497,CAR_001498,Tata Indica Vista Aqua 1.3 Quadrajet,2011,160000.0,80000.0,Diesel,Individual,Manual,Second Owner,1498,14,Low,High,0.0,Diesel_Manual,Tata,2.0,1
1498,CAR_001499,Datsun GO Plus T Option Petrol,2018,434999.0,10000.0,Petrol,Individual,Manual,First Owner,1499,7,Mid,Low,0.0,Petrol_Manual,Datsun,43.5,0
1499,CAR_001500,Mahindra XUV500 W8 4WD,2014,800000.0,156000.0,Diesel,Individual,Manual,First Owner,1500,11,Low,Very High,0.0,Diesel_Manual,Mahindra,5.13,1
1500,CAR_001501,Toyota Innova 2.5 VX (Diesel) 7 Seater BS IV,2010,819999.0,120000.0,Diesel,Individual,Manual,Second Owner,1501,15,High,High,0.0,Diesel_Manual,Toyota,6.83,1
1501,CAR_001502,Maruti Baleno Vxi,2007,4461000.0,100000.0,Petrol,Individual,Manual,Second Owner,1502,18,Low,High,0.0,Petrol_Manual,Maruti,1.63,0
1502,CAR_001503,Mercedes-Benz E-Class 220 CDI,2001,350000.0,100000.0,Diesel,Individual,Manual,Second Owner,1503,24,Mid,High,0.0,Diesel_Manual,Mercedes-Benz,3.5,1
1503,CAR_001504,Tata Indica GLS BS IV,2006,75000.0,100000.0,Petrol,Individual,Manual,Second Owner,1504,19,Low,High,0.0,Petrol_Manual,Tata,0.75,0
1504,CAR_001505,Maruti Alto LXi,2012,180000.0,60000.0,Petrol,Individual,Manual,Second Owner,1505,13,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
1505,CAR_001506,Ford Figo Aspire 1.5 TDCi Trend,2017,4461000.0,30000.0,Diesel,Individual,Manual,First Owner,1506,8,High,Low,0.0,Diesel_Manual,Ford,22.5,1
1506,CAR_001507,Maruti Swift VDI,2013,400000.0,170000.0,Diesel,Individual,Manual,First Owner,1507,12,Mid,Very High,0.0,Diesel_Manual,Maruti,2.35,1
1507,CAR_001508,Volkswagen Polo Diesel Highline 1.2L,2013,434999.0,100000.0,Diesel,Individual,Manual,Second Owner,1508,12,Mid,High,0.0,Diesel_Manual,Volkswagen,4.35,1
1508,CAR_001509,Tata Indigo GLX,2006,100000.0,110000.0,Petrol,Individual,Manual,Second Owner,1509,19,Low,High,0.0,Petrol_Manual,Tata,0.91,0
1509,CAR_001510,Maruti Swift Dzire LDI Optional,2015,4461000.0,170000.0,Diesel,Individual,Manual,Second Owner,1510,10,Mid,Very High,0.0,Diesel_Manual,Maruti,2.46,1
1510,CAR_001511,Tata Indica GLS BS IV,2009,200000.0,100000.0,Petrol,Individual,Manual,First Owner,1511,16,Low,High,0.0,Petrol_Manual,Tata,2.0,0
1511,CAR_001512,Renault Scala Diesel RxL,2014,400000.0,35000.0,Diesel,Individual,Manual,First Owner,1512,11,Mid,Medium,0.0,Diesel_Manual,Renault,11.43,1
1512,CAR_001513,Ford EcoSport 1.5 Diesel Ambiente BSIV,2018,790000.0,40000.0,Diesel,Individual,Manual,Second Owner,1513,7,High,Medium,0.0,Diesel_Manual,Ford,19.75,1
1513,CAR_001514,Renault Duster 85PS Diesel RxL Plus,2015,500000.0,90000.0,Diesel,Individual,Manual,Second Owner,1514,10,Mid,High,0.0,Diesel_Manual,Renault,5.56,1
1514,CAR_001515,Skoda Rapid 1.6 TDI Elegance,2012,275000.0,120000.0,Diesel,Individual,Manual,Second Owner,1515,13,Low,High,0.0,Diesel_Manual,Skoda,2.29,1
1515,CAR_001516,Hyundai i20 1.4 CRDi Sportz,2012,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,1516,13,Mid,High,0.0,Diesel_Manual,Hyundai,3.75,1
1516,CAR_001517,Mahindra XUV500 W11 Option AWD,2020,1400000.0,60000.0,Diesel,Dealer,Manual,First Owner,1517,5,Premium,Low,0.0,Diesel_Manual,Mahindra,56.0,1
1517,CAR_001518,Renault Duster 110PS Diesel RxL,2015,800000.0,15000.0,Diesel,Individual,Manual,First Owner,1518,10,High,Low,0.0,Diesel_Manual,Renault,53.33,1
1518,CAR_001519,Maruti Wagon R VXI,2014,315000.0,41000.0,Petrol,Dealer,Manual,First Owner,1519,11,Low,Medium,0.0,Petrol_Manual,Maruti,7.68,0
1519,CAR_001520,Maruti Baleno Alpha,2019,700000.0,16000.0,Petrol,Individual,Manual,First Owner,1520,6,High,Low,0.0,Petrol_Manual,Maruti,43.75,0
1520,CAR_001521,Mahindra Bolero SLX,2016,585000.0,51000.0,Diesel,Dealer,Manual,First Owner,1521,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,11.47,1
1521,CAR_001522,Chevrolet Sail 1.2 LT ABS,2014,350000.0,30000.0,CNG,Individual,Manual,First Owner,1522,11,Low,Low,0.0,Petrol_Manual,Chevrolet,11.67,0
1522,CAR_001523,Honda City i-DTEC SV,2014,4461000.0,80000.0,Diesel,Dealer,Manual,First Owner,1523,11,Mid,High,0.0,Diesel_Manual,Honda,6.19,1
1523,CAR_001524,Hyundai Grand i10 1.2 Kappa Asta,2017,500000.0,60000.0,Petrol,Individual,Manual,First Owner,1524,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,10.0,0
1524,CAR_001525,Hyundai Grand i10 1.2 Kappa Sportz Option,2013,345000.0,44000.0,Petrol,Dealer,Manual,First Owner,1525,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.84,0
1525,CAR_001526,Maruti Baleno Sigma 1.2,2015,500000.0,10000.0,Petrol,Individual,Manual,First Owner,1526,10,Mid,Low,0.0,Petrol_Manual,Maruti,50.0,0
1526,CAR_001527,Maruti Wagon R VXI,2016,360000.0,33000.0,Petrol,Dealer,Manual,First Owner,1527,9,Mid,Medium,0.0,Petrol_Manual,Maruti,10.91,0
1527,CAR_001528,Maruti Alto LXi BSIII,2010,210000.0,60000.0,Petrol,Individual,Manual,First Owner,1528,15,Low,Low,0.0,Petrol_Manual,Maruti,8.4,0
1528,CAR_001529,Maruti Alto K10 LXI,2014,220000.0,66000.0,LPG,Dealer,Manual,First Owner,1529,11,Low,Medium,0.0,Petrol_Manual,Maruti,3.33,0
1529,CAR_001530,Chevrolet Aveo 1.4 LT BSIV,2009,250000.0,48000.0,Petrol,Individual,Manual,First Owner,1530,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.21,0
1530,CAR_001531,Toyota Innova 2.5 G (Diesel) 8 Seater,2016,990000.0,146000.0,Diesel,Dealer,Manual,First Owner,1531,9,High,High,0.0,Diesel_Manual,Toyota,6.78,1
1531,CAR_001532,Mahindra Quanto C4,2013,220000.0,60000.0,Diesel,Individual,Manual,Fourth & Above Owner,1532,12,Low,High,0.0,Diesel_Manual,Mahindra,2.2,1
1532,CAR_001533,Jaguar XF 3.0 Litre S Premium Luxury,2012,1800000.0,50000.0,Diesel,Individual,Automatic,Second Owner,1533,13,Premium,Medium,0.0,Diesel_Automatic,Jaguar,36.0,1
1533,CAR_001534,Honda City i VTEC V,2014,530000.0,50000.0,Petrol,Individual,Manual,First Owner,1534,11,Mid,Medium,0.0,Petrol_Manual,Honda,10.6,0
1534,CAR_001535,Maruti Baleno Alpha 1.3,2016,575000.0,50000.0,Diesel,Individual,Manual,First Owner,1535,9,Mid,Medium,0.0,Diesel_Manual,Maruti,11.5,1
1535,CAR_001536,Mahindra Xylo E4,2012,400000.0,99000.0,Diesel,Dealer,Manual,First Owner,1536,13,Mid,High,0.0,Diesel_Manual,Mahindra,4.04,1
1536,CAR_001537,Honda Mobilio S i DTEC,2016,680000.0,50000.0,Diesel,Individual,Manual,First Owner,1537,9,Low,Medium,0.0,Diesel_Manual,Honda,13.6,1
1537,CAR_001538,Hyundai Santa Fe 4WD AT,2014,1000000.0,80000.0,Diesel,Individual,Automatic,First Owner,1538,11,High,High,0.0,Diesel_Automatic,Hyundai,12.5,1
1538,CAR_001539,Maruti Alto 800 LXI,2016,260000.0,45000.0,Petrol,Individual,Manual,Second Owner,1539,9,Low,Medium,0.0,Petrol_Manual,Maruti,5.78,0
1539,CAR_001540,Tata Manza Aura (ABS) Safire BS IV,2010,150000.0,50000.0,Petrol,Individual,Manual,Second Owner,1540,15,Low,Medium,0.0,Petrol_Manual,Tata,3.0,0
1540,CAR_001541,Maruti Alto LXi BSIII,2008,180000.0,70000.0,Petrol,Individual,Manual,Third Owner,1541,17,Low,Medium,0.0,Petrol_Manual,Maruti,2.57,0
1541,CAR_001542,Force One EX,2014,346000.0,37516.0,Diesel,Individual,Manual,First Owner,1542,11,Mid,Medium,0.0,Diesel_Manual,Force,9.22,1
1542,CAR_001543,Mahindra Xylo D2 BS IV,2015,450000.0,20000.0,Diesel,Individual,Manual,First Owner,1543,10,Mid,Low,0.0,Diesel_Manual,Mahindra,22.5,1
1543,CAR_001544,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2019,4461000.0,5000.0,Petrol,Individual,Manual,First Owner,1544,6,Mid,Low,0.0,Petrol_Manual,Hyundai,100.0,0
1544,CAR_001545,Hyundai Grand i10 1.2 CRDi Asta,2019,700000.0,8000.0,Diesel,Individual,Manual,First Owner,1545,6,Low,Low,0.0,Diesel_Manual,Hyundai,87.5,1
1545,CAR_001546,Maruti Alto 800 LXI,2019,280000.0,25880.0,Petrol,Individual,Manual,First Owner,1546,6,Low,Low,0.0,Petrol_Manual,Maruti,10.82,0
1546,CAR_001547,Hyundai Santro Magna AMT BSIV,2019,475000.0,10000.0,Petrol,Individual,Automatic,First Owner,1547,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,47.5,0
1547,CAR_001548,Maruti Swift Dzire VDi,2009,300000.0,50000.0,Diesel,Individual,Manual,First Owner,1548,16,Low,Medium,0.0,Diesel_Manual,Maruti,6.0,1
1548,CAR_001549,Maruti Alto K10 LXI,2014,200999.0,40000.0,Petrol,Individual,Manual,First Owner,1549,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.02,0
1549,CAR_001550,Skoda Laura 1.9 TDI MT Ambiente,2009,200000.0,60000.0,Diesel,Individual,Manual,Third Owner,1550,16,Low,High,0.0,Diesel_Manual,Skoda,2.0,1
1550,CAR_001551,Chevrolet Aveo U-VA 1.2 LS,2009,4461000.0,80000.0,Petrol,Individual,Manual,Second Owner,1551,16,Low,High,0.0,Petrol_Manual,Chevrolet,2.62,0
1551,CAR_001552,Tata Indigo LS BSII,2012,4461000.0,110000.0,Diesel,Individual,Manual,First Owner,1552,13,Low,High,0.0,Diesel_Manual,Tata,1.45,1
1552,CAR_001553,Hyundai Verna CRDi SX ABS,2009,4461000.0,136000.0,Diesel,Individual,Manual,Second Owner,1553,16,Low,High,0.0,Diesel_Manual,Hyundai,1.29,1
1553,CAR_001554,Tata New Safari DICOR 2.2 GX 4x2 BS IV,2012,320000.0,80000.0,Diesel,Individual,Manual,First Owner,1554,13,Mid,High,0.0,Diesel_Manual,Tata,4.0,1
1554,CAR_001555,Maruti Alto LXi,2009,110000.0,50000.0,Electric,Individual,Manual,Second Owner,1555,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.2,0
1555,CAR_001556,Maruti Wagon R LXI Minor,2008,300000.0,30000.0,Petrol,Individual,Manual,First Owner,1556,17,Low,Low,0.0,Petrol_Manual,Maruti,10.0,0
1556,CAR_001557,Honda City V MT,2012,700000.0,60000.0,Petrol,Individual,Manual,First Owner,1557,13,High,Low,0.0,Petrol_Manual,Honda,70.0,0
1557,CAR_001558,Tata Indica Vista Aura 1.3 Quadrajet,2009,135000.0,90000.0,Diesel,Individual,Manual,Fourth & Above Owner,1558,16,Low,Medium,0.0,Diesel_Manual,Tata,1.5,1
1558,CAR_001559,Mahindra Bolero Power Plus Plus Non AC BSIV PS,2015,500000.0,110000.0,Diesel,Individual,Manual,First Owner,1559,10,Mid,High,0.0,Diesel_Manual,Mahindra,4.55,1
1559,CAR_001560,Mahindra Scorpio VLX 2WD AIRBAG BSIV,2012,550000.0,60000.0,Diesel,Individual,Manual,Second Owner,1560,13,Mid,High,0.0,Diesel_Manual,Mahindra,4.58,1
1560,CAR_001561,Toyota Innova 2.5 Z Diesel 7 Seater BS IV,2014,1050000.0,90000.0,Diesel,Individual,Manual,Third Owner,1561,11,Premium,High,0.0,Diesel_Manual,Toyota,11.67,1
1561,CAR_001562,Honda City i VTEC V,2016,500000.0,20000.0,Petrol,Individual,Manual,First Owner,1562,9,Mid,Low,0.0,Petrol_Manual,Honda,25.0,0
1562,CAR_001563,Maruti Ritz VDi,2011,180000.0,60000.0,Petrol,Individual,Manual,Second Owner,1563,14,Low,High,0.0,Diesel_Manual,Maruti,2.0,1
1563,CAR_001564,Hyundai Santro Xing GLS,2010,170000.0,50000.0,Diesel,Individual,Manual,Second Owner,1564,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.4,0
1564,CAR_001565,Honda Brio 1.2 S Option MT,2019,4461000.0,9000.0,Petrol,Individual,Manual,First Owner,1565,6,Mid,Low,0.0,Petrol_Manual,Honda,53.89,0
1565,CAR_001566,Maruti A-Star Vxi,2011,150000.0,70000.0,Petrol,Individual,Manual,Second Owner,1566,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.14,0
1566,CAR_001567,Maruti Alto K10 VXI Airbag,2018,380000.0,25000.0,Petrol,Individual,Manual,First Owner,1567,7,Mid,Low,0.0,Petrol_Manual,Maruti,15.2,0
1567,CAR_001568,Hyundai Santro Sportz BSIV,2019,480000.0,10000.0,Petrol,Individual,Manual,First Owner,1568,6,Mid,Low,0.0,Petrol_Manual,Hyundai,48.0,0
1568,CAR_001569,Tata Indica Vista Aura 1.3 Quadrajet,2009,135000.0,90000.0,Diesel,Individual,Manual,Fourth & Above Owner,1569,16,Low,High,0.0,Diesel_Manual,Tata,1.5,1
1569,CAR_001570,Volkswagen Polo Diesel Highline 1.2L,2011,290000.0,60000.0,Diesel,Individual,Manual,Second Owner,1570,14,Low,Medium,0.0,Diesel_Manual,Volkswagen,4.83,1
1570,CAR_001571,Fiat Linea Dynamic,2013,350000.0,70000.0,Petrol,Individual,Manual,First Owner,1571,12,Mid,Medium,0.0,Petrol_Manual,Fiat,5.0,0
1571,CAR_001572,Maruti Alto 800 LXI,2014,110000.0,25000.0,Petrol,Individual,Manual,First Owner,1572,11,Low,Low,0.0,Petrol_Manual,Maruti,4.4,0
1572,CAR_001573,Hyundai Santro LE zipPlus,2003,70000.0,80000.0,Petrol,Individual,Manual,Third Owner,1573,22,Low,High,0.0,Petrol_Manual,Hyundai,0.88,0
1573,CAR_001574,Volkswagen Vento Petrol Highline AT,2011,300000.0,70000.0,Petrol,Individual,Automatic,Third Owner,1574,14,Low,Medium,0.0,Petrol_Automatic,Volkswagen,4.29,0
1574,CAR_001575,Maruti Alto 800 LX,2017,180000.0,60000.0,Petrol,Individual,Manual,First Owner,1575,8,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
1575,CAR_001576,Renault KWID RXL,2020,300000.0,20000.0,Petrol,Individual,Manual,First Owner,1576,5,Low,Low,0.0,Petrol_Manual,Renault,15.0,0
1576,CAR_001577,Maruti Baleno Delta 1.2,2017,509000.0,60000.0,Petrol,Individual,Manual,First Owner,1577,8,Mid,Medium,0.0,Petrol_Manual,Maruti,8.48,0
1577,CAR_001578,Maruti SX4 S Cross DDiS 320 Zeta,2016,650000.0,80000.0,Diesel,Individual,Manual,First Owner,1578,9,High,High,0.0,Diesel_Manual,Maruti,8.12,1
1578,CAR_001579,Maruti Swift VDI BSIV,2015,450000.0,70000.0,Diesel,Individual,Manual,Third Owner,1579,10,Low,Medium,0.0,Diesel_Manual,Maruti,6.43,1
1579,CAR_001580,Mahindra Jeep Classic,1999,170000.0,2020.0,Diesel,Individual,Manual,Second Owner,1580,26,Low,Low,0.0,Diesel_Manual,Mahindra,84.16,1
1580,CAR_001581,Mahindra Scorpio 2.6 Turbo 7 Str,2007,4461000.0,100000.0,Diesel,Individual,Manual,Second Owner,1581,18,Low,High,0.0,Diesel_Manual,Mahindra,2.1,1
1581,CAR_001582,Hyundai i20 Asta 1.4 CRDi,2013,360000.0,80000.0,Diesel,Individual,Manual,Second Owner,1582,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.5,1
1582,CAR_001583,Toyota Innova 2.5 GX (Diesel) 7 Seater BS IV,2011,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,1583,14,Mid,High,0.0,Diesel_Manual,Toyota,4.12,1
1583,CAR_001584,Maruti Alto LX,2006,105000.0,100000.0,Petrol,Individual,Manual,Third Owner,1584,19,Low,High,0.0,Petrol_Manual,Maruti,1.05,0
1584,CAR_001585,Hyundai i20 Magna,2010,270000.0,60000.0,Petrol,Individual,Manual,Third Owner,1585,15,Low,Medium,0.0,Petrol_Manual,Hyundai,4.5,0
1585,CAR_001586,Ford Figo Petrol LXI,2010,130000.0,110000.0,Petrol,Individual,Manual,First Owner,1586,15,Low,High,0.0,Petrol_Manual,Ford,1.18,0
1586,CAR_001587,Honda City i DTEC S,2014,4461000.0,90000.0,Diesel,Individual,Manual,Second Owner,1587,11,High,High,0.0,Diesel_Manual,Honda,7.78,1
1587,CAR_001588,Chevrolet Cruze LTZ,2012,400000.0,40000.0,Diesel,Individual,Manual,First Owner,1588,13,Mid,Medium,0.0,Diesel_Manual,Chevrolet,10.0,1
1588,CAR_001589,Maruti Swift ZXI ABS,2009,250000.0,60000.0,Petrol,Individual,Manual,Second Owner,1589,16,Low,Medium,0.0,Petrol_Manual,Maruti,4.17,0
1589,CAR_001590,Honda Amaze VX i-DTEC,2013,325000.0,94000.0,Diesel,Individual,Manual,Second Owner,1590,12,Mid,High,0.0,Diesel_Manual,Honda,3.46,1
1590,CAR_001591,Maruti Alto LXi,2008,69000.0,100000.0,Petrol,Individual,Manual,First Owner,1591,17,Low,Medium,0.0,Petrol_Manual,Maruti,0.69,0
1591,CAR_001592,Mahindra Scorpio S11 BSIV,2018,1380000.0,25000.0,Diesel,Individual,Manual,First Owner,1592,7,Premium,Low,0.0,Diesel_Manual,Mahindra,55.2,1
1592,CAR_001593,Maruti Zen Estilo VXI BSIII,2007,90000.0,90000.0,Petrol,Individual,Manual,First Owner,1593,18,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
1593,CAR_001594,Mahindra XUV500 W6 1.99 mHawk,2016,4461000.0,40000.0,Diesel,Individual,Manual,First Owner,1594,9,High,Medium,0.0,Diesel_Manual,Mahindra,25.0,1
1594,CAR_001595,Hyundai Accent GLE CNG,2010,145000.0,90000.0,CNG,Individual,Manual,Second Owner,1595,15,Low,High,0.0,CNG_Manual,Hyundai,1.61,0
1595,CAR_001596,Maruti Alto K10 LX,2020,250000.0,1100.0,Petrol,Individual,Manual,Fourth & Above Owner,1596,5,Low,Low,0.0,Petrol_Manual,Maruti,227.27,0
1596,CAR_001597,Maruti Swift VDI,2011,256000.0,70000.0,Diesel,Individual,Manual,Third Owner,1597,14,Low,Medium,0.0,Diesel_Manual,Maruti,3.66,1
1597,CAR_001598,Tata Indica DLS,2005,70000.0,120000.0,Diesel,Individual,Manual,First Owner,1598,20,Low,High,0.0,Diesel_Manual,Tata,0.58,1
1598,CAR_001599,Maruti Wagon R VX,2000,50000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,1599,25,Low,High,0.0,Petrol_Manual,Maruti,0.56,0
1599,CAR_001600,Renault KWID RXT,2016,240000.0,70000.0,Petrol,Individual,Manual,Second Owner,1600,9,Low,Medium,0.0,Petrol_Manual,Renault,3.43,0
1600,CAR_001601,Tata Indica Vista Aqua 1.2 Safire BSIV,2010,4461000.0,128000.0,CNG,Individual,Manual,First Owner,1601,15,Low,High,0.0,Petrol_Manual,Tata,0.76,0
1601,CAR_001602,Tata Indigo CR4,2012,130000.0,120000.0,Diesel,Individual,Manual,First Owner,1602,13,Low,Medium,0.0,Diesel_Manual,Tata,1.08,1
1602,CAR_001603,Maruti Swift 1.3 VXi,2009,4461000.0,52536.0,Petrol,Individual,Manual,First Owner,1603,16,Low,Medium,0.0,Petrol_Manual,Maruti,3.79,0
1603,CAR_001604,Honda Civic 1.8 S AT,2006,125000.0,70000.0,Petrol,Individual,Automatic,First Owner,1604,19,Low,Medium,0.0,Petrol_Automatic,Honda,1.79,0
1604,CAR_001605,Hyundai Xcent 1.1 CRDi Base,2015,420000.0,60000.0,Diesel,Individual,Manual,First Owner,1605,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.0,1
1605,CAR_001606,Mahindra NuvoSport N8,2016,525000.0,110000.0,Diesel,Individual,Manual,First Owner,1606,9,Mid,High,0.0,Diesel_Manual,Mahindra,4.77,1
1606,CAR_001607,Hyundai Santro Xing GLS,2008,125000.0,60000.0,Petrol,Individual,Manual,First Owner,1607,17,Low,High,0.0,Petrol_Manual,Hyundai,1.47,0
1607,CAR_001608,Hyundai Verna Transform CRDi VGT SX ABS,2010,325000.0,60000.0,Diesel,Individual,Manual,First Owner,1608,15,Mid,Medium,0.0,Diesel_Manual,Hyundai,2.41,1
1608,CAR_001609,Volkswagen Vento New Diesel Highline,2012,330000.0,86000.0,LPG,Individual,Manual,Fourth & Above Owner,1609,13,Mid,High,0.0,Diesel_Manual,Volkswagen,3.84,1
1609,CAR_001610,Maruti Wagon R LXI BS IV,2010,110000.0,50000.0,Petrol,Individual,Manual,First Owner,1610,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.2,0
1610,CAR_001611,Maruti S-Presso VXI Plus,2019,450000.0,1950.0,Petrol,Individual,Manual,First Owner,1611,6,Mid,Low,0.0,Petrol_Manual,Maruti,230.77,0
1611,CAR_001612,Maruti Swift 1.3 VXi,2005,85000.0,118400.0,Petrol,Individual,Manual,First Owner,1612,20,Low,Medium,0.0,Petrol_Manual,Maruti,0.72,0
1612,CAR_001613,Honda Civic 1.8 S MT,2007,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,1613,18,Low,Medium,0.0,Petrol_Manual,Honda,3.29,0
1613,CAR_001614,Hyundai i10 Magna,2012,250000.0,60000.0,Petrol,Individual,Manual,Second Owner,1614,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.17,0
1614,CAR_001615,Ford Ecosport 1.5 DV5 MT Titanium,2015,650000.0,68000.0,Diesel,Individual,Manual,First Owner,1615,10,High,Medium,0.0,Diesel_Manual,Ford,9.56,1
1615,CAR_001616,Maruti Swift Dzire VDI,2017,600000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,1616,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
1616,CAR_001617,Toyota Etios Liva 1.2 VX,2017,650000.0,6480.0,Petrol,Trustmark Dealer,Manual,First Owner,1617,8,High,Low,0.0,Petrol_Manual,Toyota,100.31,0
1617,CAR_001618,Maruti Vitara Brezza VDi,2017,750000.0,32077.0,Diesel,Trustmark Dealer,Manual,First Owner,1618,8,High,Medium,0.0,Diesel_Manual,Maruti,23.38,1
1618,CAR_001619,Toyota Yaris G,2018,4461000.0,19107.0,Petrol,Trustmark Dealer,Manual,First Owner,1619,7,Low,Low,0.0,Petrol_Manual,Toyota,44.49,0
1619,CAR_001620,Hyundai EON Era Plus Option,2017,315000.0,18469.0,Petrol,Trustmark Dealer,Manual,First Owner,1620,8,Mid,Low,0.0,Petrol_Manual,Hyundai,17.06,0
1620,CAR_001621,Maruti Swift LXI Option,2015,415000.0,28217.0,Petrol,Trustmark Dealer,Manual,First Owner,1621,10,Mid,Low,0.0,Petrol_Manual,Maruti,14.71,0
1621,CAR_001622,Maruti Ciaz ZDi Plus,2016,640000.0,72787.0,Diesel,Trustmark Dealer,Manual,First Owner,1622,9,High,High,0.0,Diesel_Manual,Maruti,8.79,1
1622,CAR_001623,Maruti Ciaz 1.4 Alpha,2017,780000.0,31063.0,Petrol,Trustmark Dealer,Manual,First Owner,1623,8,High,Medium,0.0,Petrol_Manual,Maruti,25.11,0
1623,CAR_001624,Toyota Fortuner 4x2 AT,2017,2595000.0,79641.0,Diesel,Trustmark Dealer,Automatic,First Owner,1624,8,Low,High,0.0,Diesel_Automatic,Toyota,32.58,1
1624,CAR_001625,Hyundai Creta 1.6 CRDi SX,2015,850000.0,58692.0,Diesel,Trustmark Dealer,Manual,First Owner,1625,10,High,Medium,0.0,Diesel_Manual,Hyundai,14.48,1
1625,CAR_001626,Hyundai Creta 1.6 CRDi SX,2015,900000.0,54784.0,Diesel,Trustmark Dealer,Manual,First Owner,1626,10,High,Medium,0.0,Diesel_Manual,Hyundai,16.43,1
1626,CAR_001627,Toyota Corolla Altis GL MT,2016,1150000.0,64156.0,Petrol,Trustmark Dealer,Manual,First Owner,1627,9,Premium,Medium,0.0,Petrol_Manual,Toyota,17.93,0
1627,CAR_001628,Maruti Esteem Lxi - BSIII,2007,75000.0,54000.0,Petrol,Individual,Manual,First Owner,1628,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.39,0
1628,CAR_001629,Volkswagen Vento Diesel Comfortline,2012,215000.0,60000.0,Diesel,Individual,Manual,First Owner,1629,13,Low,High,0.0,Diesel_Manual,Volkswagen,2.22,1
1629,CAR_001630,Maruti SX4 ZXI AT,2009,130000.0,120000.0,Petrol,Individual,Automatic,Second Owner,1630,16,Low,High,0.0,Petrol_Automatic,Maruti,1.08,0
1630,CAR_001631,Hyundai Santro Magna AMT BSIV,2019,400000.0,16000.0,Petrol,Individual,Automatic,First Owner,1631,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,25.0,0
1631,CAR_001632,Maruti Swift Dzire AMT ZXI,2019,550000.0,60000.0,Petrol,Individual,Automatic,First Owner,1632,6,Mid,Medium,0.0,Petrol_Automatic,Maruti,9.17,0
1632,CAR_001633,Chevrolet Beat LT,2016,245000.0,52000.0,Electric,Dealer,Manual,First Owner,1633,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,4.71,0
1633,CAR_001634,Maruti Swift VDI BSIV,2010,320000.0,110000.0,Diesel,Individual,Manual,Fourth & Above Owner,1634,15,Mid,High,0.0,Diesel_Manual,Maruti,2.91,1
1634,CAR_001635,Mahindra TUV 300 T8 AMT,2016,4461000.0,60000.0,Diesel,Individual,Automatic,Second Owner,1635,9,High,Medium,0.0,Diesel_Automatic,Mahindra,12.17,1
1635,CAR_001636,Ford Fusion 1.6 Duratec Petrol,2005,120000.0,100000.0,Petrol,Individual,Manual,Third Owner,1636,20,Low,High,0.0,Petrol_Manual,Ford,1.2,0
1636,CAR_001637,Honda City i-VTEC VX,2017,950000.0,35000.0,Petrol,Individual,Manual,First Owner,1637,8,High,Medium,0.0,Petrol_Manual,Honda,27.14,0
1637,CAR_001638,Maruti Wagon R VXI BS IV,2017,350000.0,48000.0,Petrol,Individual,Manual,Third Owner,1638,8,Mid,Medium,0.0,Petrol_Manual,Maruti,7.29,0
1638,CAR_001639,Maruti Esteem Vxi,2005,100000.0,20000.0,Petrol,Individual,Manual,Second Owner,1639,20,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
1639,CAR_001640,Maruti Wagon R VXI BS IV,2017,350000.0,60000.0,Petrol,Individual,Manual,First Owner,1640,8,Low,Low,0.0,Petrol_Manual,Maruti,11.67,0
1640,CAR_001641,Maruti Wagon R VXI BSIII,2005,80000.0,40000.0,Petrol,Individual,Manual,Second Owner,1641,20,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
1641,CAR_001642,Hyundai Verna 1.6 VTVT,2013,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,1642,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.17,0
1642,CAR_001643,Hyundai EON D Lite Plus,2012,160000.0,70000.0,Petrol,Individual,Manual,Second Owner,1643,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.29,0
1643,CAR_001644,Tata Sumo Victa EX 7/9 Str BSII,2009,4461000.0,120000.0,Diesel,Individual,Manual,First Owner,1644,16,Low,High,0.0,Diesel_Manual,Tata,1.08,1
1644,CAR_001645,Chevrolet Sail Hatchback 1.2 LS,2016,250000.0,30000.0,Petrol,Individual,Manual,First Owner,1645,9,Low,Low,0.0,Petrol_Manual,Chevrolet,8.33,0
1645,CAR_001646,Tata New Safari DICOR 2.2 GX 4x2 BS IV,2012,270000.0,80000.0,Diesel,Individual,Manual,Second Owner,1646,13,Low,Medium,0.0,Diesel_Manual,Tata,3.38,1
1646,CAR_001647,BMW X1 sDrive 20d xLine,2019,2600000.0,9500.0,Diesel,Individual,Automatic,First Owner,1647,6,Premium,Low,0.0,Diesel_Automatic,BMW,273.68,1
1647,CAR_001648,Hyundai EON Era Plus,2013,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,1648,12,Low,High,0.0,Petrol_Manual,Hyundai,2.0,0
1648,CAR_001649,Hyundai EON Magna Plus,2016,229999.0,60000.0,Petrol,Individual,Manual,First Owner,1649,9,Low,High,0.0,Petrol_Manual,Hyundai,2.87,0
1649,CAR_001650,Mahindra Bolero Power Plus ZLX,2017,750000.0,29000.0,Diesel,Individual,Manual,First Owner,1650,8,High,Low,0.0,Diesel_Manual,Mahindra,25.86,1
1650,CAR_001651,Chevrolet Optra Magnum 2.0 LS,2011,4461000.0,120000.0,CNG,Individual,Manual,First Owner,1651,14,Low,High,0.0,Diesel_Manual,Chevrolet,1.62,1
1651,CAR_001652,Toyota Etios Cross 1.2L G,2015,320000.0,60000.0,Petrol,Individual,Manual,First Owner,1652,10,Mid,Medium,0.0,Petrol_Manual,Toyota,5.33,0
1652,CAR_001653,Mahindra Scorpio S7 140 BSIV,2018,1100000.0,20000.0,LPG,Individual,Manual,First Owner,1653,7,Premium,Low,0.0,Diesel_Manual,Mahindra,55.0,1
1653,CAR_001654,Datsun RediGO S,2016,270000.0,22000.0,Petrol,Individual,Manual,First Owner,1654,9,Low,Low,0.0,Petrol_Manual,Datsun,12.27,0
1654,CAR_001655,Skoda Fabia 1.2 MPI Ambition Plus,2012,300000.0,50000.0,Petrol,Individual,Manual,First Owner,1655,13,Low,Medium,0.0,Petrol_Manual,Skoda,6.0,0
1655,CAR_001656,Chevrolet Beat Diesel LT,2012,200000.0,50000.0,Electric,Individual,Manual,First Owner,1656,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,4.0,1
1656,CAR_001657,Maruti Zen Estilo 1.1 LXI BSIII,2007,120000.0,81366.0,Petrol,Individual,Manual,First Owner,1657,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.47,0
1657,CAR_001658,Maruti Esteem Vxi - BSIII,2006,4461000.0,110000.0,Petrol,Individual,Manual,First Owner,1658,19,Low,High,0.0,Petrol_Manual,Maruti,0.86,0
1658,CAR_001659,Volkswagen Polo Petrol Comfortline 1.2L,2013,400000.0,100000.0,Petrol,Individual,Manual,First Owner,1659,12,Mid,High,0.0,Petrol_Manual,Volkswagen,4.0,0
1659,CAR_001660,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2006,4461000.0,300000.0,Diesel,Individual,Manual,First Owner,1660,19,Low,Very High,0.0,Diesel_Manual,Toyota,0.77,1
1660,CAR_001661,Mahindra KUV 100 mFALCON D75 K6,2016,430000.0,20000.0,Diesel,Individual,Manual,First Owner,1661,9,Mid,Low,0.0,Diesel_Manual,Mahindra,21.5,1
1661,CAR_001662,Renault Fluence 1.5,2012,300000.0,90000.0,Petrol,Individual,Manual,First Owner,1662,13,Low,High,0.0,Diesel_Manual,Renault,3.33,1
1662,CAR_001663,Fiat Linea Dynamic,2013,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,1663,12,Mid,Medium,0.0,Petrol_Manual,Fiat,5.0,0
1663,CAR_001664,Renault Duster 110PS Diesel RxZ,2013,450000.0,120000.0,Diesel,Individual,Manual,First Owner,1664,12,Mid,High,0.0,Diesel_Manual,Renault,3.75,1
1664,CAR_001665,Renault KWID RXT,2016,200000.0,53000.0,Petrol,Individual,Manual,Second Owner,1665,9,Low,Medium,0.0,Petrol_Manual,Renault,3.77,0
1665,CAR_001666,Maruti Alto 800 LXI,2017,250000.0,60000.0,Petrol,Individual,Manual,First Owner,1666,8,Low,Low,0.0,Petrol_Manual,Maruti,8.33,0
1666,CAR_001667,Maruti Swift Dzire 1.2 Vxi BSIV,2012,368000.0,90000.0,Petrol,Individual,Manual,Second Owner,1667,13,Mid,High,0.0,Petrol_Manual,Maruti,4.09,0
1667,CAR_001668,Maruti Swift Dzire 1.2 Vxi BSIV,2012,368000.0,90000.0,Petrol,Individual,Manual,Second Owner,1668,13,Mid,High,0.0,Petrol_Manual,Maruti,4.09,0
1668,CAR_001669,Toyota Innova 2.5 GX (Diesel) 7 Seater,2014,650000.0,244000.0,Diesel,Individual,Manual,First Owner,1669,11,High,Medium,0.0,Diesel_Manual,Toyota,2.66,1
1669,CAR_001670,Mahindra Jeep CL 500 MDI,1997,150000.0,120000.0,Diesel,Individual,Manual,Third Owner,1670,28,Low,High,0.0,Diesel_Manual,Mahindra,1.25,1
1670,CAR_001671,Maruti Wagon R LXI DUO BS IV,2012,180000.0,60000.0,LPG,Individual,Manual,First Owner,1671,13,Low,Medium,0.0,LPG_Manual,Maruti,3.0,0
1671,CAR_001672,Hyundai Grand i10 1.2 Kappa Sportz Option,2017,525000.0,15000.0,Petrol,Individual,Manual,First Owner,1672,8,Mid,Low,0.0,Petrol_Manual,Hyundai,35.0,0
1672,CAR_001673,Ford Endeavour 4x4 XLT,2005,350000.0,150000.0,Diesel,Individual,Manual,Second Owner,1673,20,Mid,High,0.0,Diesel_Manual,Ford,2.33,1
1673,CAR_001674,Mahindra TUV 300 Plus P4,2018,850000.0,30000.0,Diesel,Individual,Manual,Second Owner,1674,7,High,Low,0.0,Diesel_Manual,Mahindra,28.33,1
1674,CAR_001675,Volkswagen Jetta 2.0 TDI Comfortline,2011,350000.0,312000.0,Diesel,Individual,Manual,Third Owner,1675,14,Mid,Very High,0.0,Diesel_Manual,Volkswagen,1.12,1
1675,CAR_001676,Maruti Esteem Lxi - BSIII,2007,75000.0,54000.0,Petrol,Individual,Manual,First Owner,1676,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.39,0
1676,CAR_001677,Maruti Swift 1.3 VXi,2005,160000.0,67000.0,Petrol,Individual,Manual,First Owner,1677,20,Low,Medium,0.0,Petrol_Manual,Maruti,2.39,0
1677,CAR_001678,Maruti Swift LDI Optional,2017,500000.0,40000.0,Diesel,Individual,Manual,First Owner,1678,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.5,1
1678,CAR_001679,Hyundai Santro Xing GLS,2008,130000.0,100000.0,Petrol,Individual,Manual,First Owner,1679,17,Low,High,0.0,Petrol_Manual,Hyundai,1.3,0
1679,CAR_001680,Tata Zest Quadrajet 1.3 75PS XE,2015,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,1680,10,Mid,Medium,0.0,Diesel_Manual,Tata,3.5,1
1680,CAR_001681,Maruti Wagon R LXI,2005,70000.0,80000.0,CNG,Individual,Manual,First Owner,1681,20,Low,High,0.0,Petrol_Manual,Maruti,0.88,0
1681,CAR_001682,Toyota Innova 2.5 G (Diesel) 7 Seater,2014,750000.0,145000.0,Diesel,Individual,Manual,Second Owner,1682,11,High,High,0.0,Diesel_Manual,Toyota,5.17,1
1682,CAR_001683,Honda Amaze VX i-DTEC,2013,450000.0,53000.0,Diesel,Individual,Manual,First Owner,1683,12,Mid,Medium,0.0,Diesel_Manual,Honda,8.49,1
1683,CAR_001684,Maruti Swift Dzire LDI,2015,430000.0,73000.0,Diesel,Individual,Manual,Third Owner,1684,10,Mid,High,0.0,Diesel_Manual,Maruti,5.89,1
1684,CAR_001685,Maruti SX4 VDI,2011,4461000.0,90000.0,Diesel,Individual,Manual,Third Owner,1685,14,Low,High,0.0,Diesel_Manual,Maruti,2.89,1
1685,CAR_001686,Maruti Swift VDI,2013,370000.0,80000.0,Diesel,Individual,Manual,Third Owner,1686,12,Mid,High,0.0,Diesel_Manual,Maruti,4.62,1
1686,CAR_001687,Chevrolet Aveo 1.4 CNG,2011,229999.0,100000.0,CNG,Individual,Manual,First Owner,1687,14,Low,High,0.0,CNG_Manual,Chevrolet,2.3,0
1687,CAR_001688,Ford Figo Diesel EXI,2011,140000.0,90000.0,Diesel,Individual,Manual,Third Owner,1688,14,Low,High,0.0,Diesel_Manual,Ford,1.56,1
1688,CAR_001689,Toyota Innova 2.5 G (Diesel) 7 Seater,2013,700000.0,200000.0,LPG,Individual,Manual,First Owner,1689,12,High,Very High,0.0,Diesel_Manual,Toyota,3.5,1
1689,CAR_001690,Hyundai Elite i20 Magna Plus BSIV,2020,545000.0,7300.0,Petrol,Individual,Manual,First Owner,1690,5,Mid,Low,0.0,Petrol_Manual,Hyundai,74.66,0
1690,CAR_001691,Datsun RediGO T Option,2017,200000.0,30000.0,Petrol,Individual,Manual,First Owner,1691,8,Low,Low,0.0,Petrol_Manual,Datsun,6.67,0
1691,CAR_001692,Tata Safari Storme VX,2015,500000.0,60000.0,Diesel,Individual,Manual,First Owner,1692,10,Mid,High,0.0,Diesel_Manual,Tata,4.17,1
1692,CAR_001693,Maruti Alto LXi,2008,70000.0,90000.0,Petrol,Individual,Manual,Second Owner,1693,17,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
1693,CAR_001694,Mahindra Thar 4X4,2013,580000.0,25000.0,Electric,Individual,Manual,Third Owner,1694,12,Mid,Low,0.0,Diesel_Manual,Mahindra,23.2,1
1694,CAR_001695,Maruti Eeco 5 STR With AC Plus HTR CNG,2012,160000.0,70000.0,CNG,Individual,Manual,Second Owner,1695,13,Low,Medium,0.0,CNG_Manual,Maruti,2.29,0
1695,CAR_001696,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2018,450000.0,6000.0,Petrol,Individual,Manual,First Owner,1696,7,Mid,Low,0.0,Petrol_Manual,Hyundai,75.0,0
1696,CAR_001697,Maruti Alto LXi,2010,160000.0,60000.0,Petrol,Individual,Manual,First Owner,1697,15,Low,Medium,0.0,Petrol_Manual,Maruti,4.44,0
1697,CAR_001698,Maruti Ertiga SHVS ZDI Plus,2016,760000.0,49000.0,Petrol,Individual,Manual,First Owner,1698,9,Low,Medium,0.0,Diesel_Manual,Maruti,15.51,1
1698,CAR_001699,Hyundai EON Magna Optional,2015,300000.0,22000.0,Petrol,Individual,Manual,First Owner,1699,10,Low,Low,0.0,Petrol_Manual,Hyundai,13.64,0
1699,CAR_001700,Maruti Omni 5 Str STD LPG,1998,50000.0,35000.0,LPG,Individual,Manual,Second Owner,1700,27,Low,Medium,0.0,LPG_Manual,Maruti,1.43,0
1700,CAR_001701,Mahindra Bolero 2011-2019 SLX,2019,800000.0,24000.0,Diesel,Individual,Manual,First Owner,1701,6,High,Low,0.0,Diesel_Manual,Mahindra,33.33,1
1701,CAR_001702,Maruti Swift Dzire VDI,2018,680000.0,60000.0,Diesel,Individual,Manual,First Owner,1702,7,High,Low,0.0,Diesel_Manual,Maruti,34.0,1
1702,CAR_001703,Maruti 800 DUO AC LPG,2009,85000.0,50000.0,LPG,Individual,Manual,Second Owner,1703,16,Low,Medium,0.0,LPG_Manual,Maruti,1.7,0
1703,CAR_001704,Toyota Innova Crysta 2.4 G MT BSIV,2016,1300000.0,70000.0,Diesel,Individual,Manual,First Owner,1704,9,Low,Medium,0.0,Diesel_Manual,Toyota,18.57,1
1704,CAR_001705,Maruti Swift VXI BSIV,2016,500000.0,60000.0,Petrol,Individual,Manual,First Owner,1705,9,Mid,Medium,0.0,Petrol_Manual,Maruti,8.33,0
1705,CAR_001706,Audi Q5 2.0 TDI,2015,3500000.0,35000.0,Diesel,Individual,Automatic,First Owner,1706,10,Premium,Medium,0.0,Diesel_Automatic,Audi,100.0,1
1706,CAR_001707,Hyundai Accent GLX,2008,145000.0,60000.0,Petrol,Individual,Manual,First Owner,1707,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.42,0
1707,CAR_001708,Mahindra Quanto C6,2013,300000.0,60000.0,Diesel,Individual,Manual,First Owner,1708,12,Low,Medium,0.0,Diesel_Manual,Mahindra,5.0,1
1708,CAR_001709,Maruti Swift VDI Optional,2017,600000.0,60000.0,Diesel,Individual,Manual,Second Owner,1709,8,Mid,Medium,0.0,Diesel_Manual,Maruti,10.0,1
1709,CAR_001710,Maruti Swift Dzire VXI,2015,400000.0,70000.0,Petrol,Individual,Manual,Second Owner,1710,10,Mid,Medium,0.0,Petrol_Manual,Maruti,5.71,0
1710,CAR_001711,Hyundai Santro Xing XK eRLX EuroIII,2007,130000.0,135000.0,Petrol,Individual,Manual,Second Owner,1711,18,Low,Medium,0.0,Petrol_Manual,Hyundai,0.96,0
1711,CAR_001712,Maruti 800 AC BSII,2001,45000.0,72539.0,Petrol,Individual,Manual,Second Owner,1712,24,Low,High,0.0,Petrol_Manual,Maruti,0.62,0
1712,CAR_001713,Maruti Baleno Zeta 1.2,2017,641000.0,25000.0,Petrol,Individual,Manual,First Owner,1713,8,High,Low,0.0,Petrol_Manual,Maruti,25.64,0
1713,CAR_001714,Maruti Swift Dzire VDI,2015,650000.0,50000.0,Diesel,Individual,Manual,First Owner,1714,10,High,Medium,0.0,Diesel_Manual,Maruti,13.0,1
1714,CAR_001715,Ford Freestyle Titanium Diesel,2020,784000.0,101.0,Diesel,Dealer,Manual,Test Drive Car,1715,5,High,Low,0.0,Diesel_Manual,Ford,7762.38,1
1715,CAR_001716,Ford Figo Titanium,2020,635000.0,101.0,Petrol,Dealer,Manual,Test Drive Car,1716,5,High,Low,0.0,Petrol_Manual,Ford,6287.13,0
1716,CAR_001717,Ford Ecosport 1.5 Diesel Titanium,2020,1000000.0,101.0,Diesel,Dealer,Manual,Test Drive Car,1717,5,High,Low,0.0,Diesel_Manual,Ford,9900.99,1
1717,CAR_001718,Ford Figo 1.5D Titanium Opt MT,2015,495000.0,52328.0,Diesel,Dealer,Manual,First Owner,1718,10,Mid,Medium,0.0,Diesel_Manual,Ford,9.46,1
1718,CAR_001719,Hyundai Santro Xing GLS,2013,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,1719,12,Mid,Low,0.0,Petrol_Manual,Hyundai,10.8,0
1719,CAR_001720,Ford Endeavour 3.2 Titanium AT 4X4,2016,2100000.0,91505.0,Diesel,Dealer,Automatic,First Owner,1720,9,Low,High,0.0,Diesel_Automatic,Ford,22.95,1
1720,CAR_001721,Maruti Alto 800 LXI,2018,200000.0,35000.0,Petrol,Individual,Manual,First Owner,1721,7,Low,Medium,0.0,Petrol_Manual,Maruti,5.71,0
1721,CAR_001722,Maruti Swift Dzire VDI,2015,320000.0,60000.0,Diesel,Individual,Manual,First Owner,1722,10,Mid,Medium,0.0,Diesel_Manual,Maruti,5.33,1
1722,CAR_001723,Hyundai EON Magna Optional,2015,300000.0,20500.0,Diesel,Individual,Manual,First Owner,1723,10,Low,Low,0.0,Petrol_Manual,Hyundai,14.63,0
1723,CAR_001724,Tata Indica Vista Quadrajet LX,2013,4461000.0,200000.0,CNG,Individual,Manual,First Owner,1724,12,Low,Very High,0.0,Diesel_Manual,Tata,0.9,1
1724,CAR_001725,Toyota Innova 2.5 V Diesel 8-seater,2008,500000.0,154000.0,Diesel,Individual,Manual,Third Owner,1725,17,Mid,Very High,0.0,Diesel_Manual,Toyota,3.25,1
1725,CAR_001726,Toyota Etios VX,2011,305000.0,120000.0,Petrol,Individual,Manual,Second Owner,1726,14,Mid,Medium,0.0,Petrol_Manual,Toyota,2.54,0
1726,CAR_001727,Maruti 800 AC,2009,125000.0,50000.0,Petrol,Individual,Manual,Fourth & Above Owner,1727,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
1727,CAR_001728,Maruti Swift Dzire ZDI,2015,800000.0,135000.0,Diesel,Individual,Manual,First Owner,1728,10,Low,High,0.0,Diesel_Manual,Maruti,5.93,1
1728,CAR_001729,Mahindra Scorpio SLE BSIV,2009,420000.0,100000.0,Diesel,Individual,Manual,First Owner,1729,16,Mid,Medium,0.0,Diesel_Manual,Mahindra,4.2,1
1729,CAR_001730,Maruti Vitara Brezza ZDi Plus,2017,875000.0,40000.0,Diesel,Individual,Manual,First Owner,1730,8,High,Medium,0.0,Diesel_Manual,Maruti,21.88,1
1730,CAR_001731,Maruti Swift Dzire ZDI,2014,580000.0,70000.0,Diesel,Individual,Manual,First Owner,1731,11,Mid,Medium,0.0,Diesel_Manual,Maruti,8.29,1
1731,CAR_001732,Maruti Swift Ldi BSIV,2011,350000.0,133000.0,Diesel,Individual,Manual,Second Owner,1732,14,Mid,High,0.0,Diesel_Manual,Maruti,2.63,1
1732,CAR_001733,Hyundai Accent GLS,2004,120000.0,110000.0,Petrol,Individual,Manual,Second Owner,1733,21,Low,High,0.0,Petrol_Manual,Hyundai,1.09,0
1733,CAR_001734,Tata Indica Vista TDI LS,2013,150000.0,180000.0,Diesel,Individual,Manual,Third Owner,1734,12,Low,Very High,0.0,Diesel_Manual,Tata,0.83,1
1734,CAR_001735,Renault Duster 110PS Diesel RxZ,2013,650000.0,120000.0,Diesel,Individual,Manual,First Owner,1735,12,High,High,0.0,Diesel_Manual,Renault,5.42,1
1735,CAR_001736,Maruti Alto K10 2010-2014 VXI,2010,225000.0,80000.0,Petrol,Individual,Manual,Second Owner,1736,15,Low,High,0.0,Petrol_Manual,Maruti,2.81,0
1736,CAR_001737,Honda Amaze S AT i-Vtech,2014,400000.0,15000.0,Petrol,Individual,Automatic,First Owner,1737,11,Mid,Low,0.0,Petrol_Automatic,Honda,26.67,0
1737,CAR_001738,Hyundai Getz GLS,2005,250000.0,70000.0,Petrol,Individual,Manual,First Owner,1738,20,Low,Medium,0.0,Petrol_Manual,Hyundai,3.57,0
1738,CAR_001739,Hyundai Santro Xing GL Plus,2008,120000.0,41723.0,Petrol,Individual,Manual,Second Owner,1739,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.88,0
1739,CAR_001740,Maruti SX4 Vxi BSIV,2010,250000.0,70000.0,Petrol,Individual,Manual,First Owner,1740,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.57,0
1740,CAR_001741,Maruti Alto K10 LXI,2012,175000.0,60000.0,LPG,Individual,Manual,Second Owner,1741,13,Low,Low,0.0,Petrol_Manual,Maruti,8.75,0
1741,CAR_001742,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,1742,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
1742,CAR_001743,Maruti Ignis 1.3 Delta,2018,590000.0,26350.0,Electric,Dealer,Manual,First Owner,1743,7,Mid,Medium,0.0,Diesel_Manual,Maruti,22.39,1
1743,CAR_001744,Tata Indigo CS LX (TDI) BS-III,2015,4461000.0,68745.0,Diesel,Dealer,Manual,First Owner,1744,10,Low,Medium,0.0,Diesel_Manual,Tata,3.27,1
1744,CAR_001745,Mahindra XUV500 W6 2WD,2016,900000.0,47000.0,Diesel,Individual,Manual,First Owner,1745,9,High,Medium,0.0,Diesel_Manual,Mahindra,19.15,1
1745,CAR_001746,Maruti Alto K10 VXI AGS,2017,375000.0,27289.0,Petrol,Dealer,Automatic,First Owner,1746,8,Mid,Low,0.0,Petrol_Automatic,Maruti,13.74,0
1746,CAR_001747,Hyundai EON Era Plus,2016,320000.0,24662.0,Petrol,Dealer,Manual,First Owner,1747,9,Mid,Low,0.0,Petrol_Manual,Hyundai,12.98,0
1747,CAR_001748,Maruti Swift Dzire VDI,2013,525000.0,37000.0,Diesel,Dealer,Manual,First Owner,1748,12,Mid,Medium,0.0,Diesel_Manual,Maruti,14.19,1
1748,CAR_001749,Nissan Evalia XV,2014,650000.0,28245.0,Diesel,Dealer,Manual,First Owner,1749,11,High,Low,0.0,Diesel_Manual,Nissan,23.01,1
1749,CAR_001750,Hyundai Grand i10 1.2 CRDi Sportz Option,2017,575000.0,27005.0,Diesel,Dealer,Manual,First Owner,1750,8,Mid,Low,0.0,Diesel_Manual,Hyundai,21.29,1
1750,CAR_001751,Hyundai i10 Magna,2014,355000.0,39227.0,Petrol,Dealer,Manual,First Owner,1751,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.05,0
1751,CAR_001752,Mahindra KUV 100 D75 K4 Plus 5Str,2016,470000.0,31367.0,Diesel,Dealer,Manual,First Owner,1752,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,14.98,1
1752,CAR_001753,Maruti Wagon R LXI,2008,221000.0,35008.0,Petrol,Dealer,Manual,First Owner,1753,17,Low,Medium,0.0,Petrol_Manual,Maruti,6.31,0
1753,CAR_001754,Hyundai Santro Xing GLS,2009,195000.0,27005.0,Petrol,Dealer,Manual,First Owner,1754,16,Low,Low,0.0,Petrol_Manual,Hyundai,7.22,0
1754,CAR_001755,Hyundai Santro Xing XG,2004,90000.0,100005.0,Petrol,Dealer,Manual,Second Owner,1755,21,Low,Medium,0.0,Petrol_Manual,Hyundai,0.9,0
1755,CAR_001756,Hyundai Verna SX Diesel,2013,580000.0,45264.0,Diesel,Dealer,Manual,First Owner,1756,12,Low,Medium,0.0,Diesel_Manual,Hyundai,12.81,1
1756,CAR_001757,Maruti Eeco 7 Seater Standard BSIV,2014,325000.0,39093.0,Petrol,Dealer,Manual,First Owner,1757,11,Mid,Medium,0.0,Petrol_Manual,Maruti,8.31,0
1757,CAR_001758,Honda City Corporate Edition,2013,495000.0,45241.0,Petrol,Dealer,Manual,First Owner,1758,12,Mid,Medium,0.0,Petrol_Manual,Honda,10.94,0
1758,CAR_001759,Hyundai EON Era,2014,290000.0,60000.0,Petrol,Dealer,Manual,First Owner,1759,11,Low,Medium,0.0,Petrol_Manual,Hyundai,24.17,0
1759,CAR_001760,Datsun RediGO 1.0 T Option,2018,375000.0,2769.0,Petrol,Dealer,Manual,First Owner,1760,7,Mid,Low,0.0,Petrol_Manual,Datsun,135.43,0
1760,CAR_001761,Tata Indica Vista TDI LS,2011,211000.0,43128.0,Diesel,Dealer,Manual,First Owner,1761,14,Low,Medium,0.0,Diesel_Manual,Tata,4.89,1
1761,CAR_001762,Hyundai i20 Asta,2010,270000.0,110000.0,Petrol,Individual,Manual,Second Owner,1762,15,Low,Medium,0.0,Petrol_Manual,Hyundai,2.45,0
1762,CAR_001763,Maruti Alto LXI,2004,70000.0,60000.0,Petrol,Individual,Manual,Second Owner,1763,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.4,0
1763,CAR_001764,Hyundai i10 Magna 1.2,2010,225000.0,60000.0,Petrol,Individual,Manual,Second Owner,1764,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.75,0
1764,CAR_001765,Mahindra Scorpio LX,2014,420000.0,70000.0,Diesel,Individual,Manual,Third Owner,1765,11,Mid,Medium,0.0,Diesel_Manual,Mahindra,6.0,1
1765,CAR_001766,Chevrolet Sail Hatchback 1.2 LS,2015,280000.0,60000.0,Petrol,Individual,Manual,First Owner,1766,10,Low,Low,0.0,Petrol_Manual,Chevrolet,12.58,0
1766,CAR_001767,Hyundai i20 Sportz 1.2,2013,400000.0,59213.0,Petrol,Individual,Manual,First Owner,1767,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.76,0
1767,CAR_001768,Mahindra XUV500 W6 2WD,2015,850000.0,55000.0,Diesel,Individual,Manual,First Owner,1768,10,High,Medium,0.0,Diesel_Manual,Mahindra,15.45,1
1768,CAR_001769,Ford Fiesta Titanium 1.5 TDCi,2011,550000.0,60000.0,Diesel,Individual,Manual,First Owner,1769,14,Mid,Medium,0.0,Diesel_Manual,Ford,6.88,1
1769,CAR_001770,Ford Fiesta Titanium 1.5 TDCi,2011,550000.0,80000.0,Diesel,Individual,Manual,First Owner,1770,14,Mid,Medium,0.0,Diesel_Manual,Ford,6.88,1
1770,CAR_001771,Renault Captur 1.5 Diesel RXT Mono,2018,1000000.0,40000.0,Diesel,Individual,Manual,First Owner,1771,7,High,Medium,0.0,Diesel_Manual,Renault,25.0,1
1771,CAR_001772,Mahindra Xylo D2 Maxx,2014,350000.0,170000.0,CNG,Individual,Manual,First Owner,1772,11,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.06,1
1772,CAR_001773,Tata New Safari DICOR 2.2 EX 4x2,2011,350000.0,170000.0,Diesel,Individual,Manual,Second Owner,1773,14,Mid,Very High,0.0,Diesel_Manual,Tata,2.06,1
1773,CAR_001774,Honda Civic 1.8 S MT,2007,170000.0,70000.0,Petrol,Individual,Manual,Third Owner,1774,18,Low,Medium,0.0,Petrol_Manual,Honda,2.43,0
1774,CAR_001775,Ford Aspire Titanium BSIV,2020,828999.0,1010.0,Petrol,Dealer,Manual,Test Drive Car,1775,5,High,Low,0.0,Petrol_Manual,Ford,820.79,0
1775,CAR_001776,Ford EcoSport 1.5 Ti VCT MT Titanium BE BSIV,2020,1119000.0,60000.0,Petrol,Dealer,Manual,Test Drive Car,1776,5,Premium,Low,0.0,Petrol_Manual,Ford,1107.92,0
1776,CAR_001777,Ford Figo Titanium,2020,746000.0,1111.0,Petrol,Dealer,Manual,Test Drive Car,1777,5,High,Low,0.0,Petrol_Manual,Ford,671.47,0
1777,CAR_001778,Ford Ecosport 1.5 Petrol Trend,2020,1030000.0,60000.0,Petrol,Dealer,Manual,Test Drive Car,1778,5,Premium,Low,0.0,Petrol_Manual,Ford,1019.8,0
1778,CAR_001779,Ford EcoSport 1.5 TDCi Titanium Plus BSIV,2020,4461000.0,1010.0,Diesel,Dealer,Manual,Test Drive Car,1779,5,Premium,Low,0.0,Diesel_Manual,Ford,1320.79,1
1779,CAR_001780,Ford Freestyle Titanium,2020,811999.0,1010.0,Petrol,Dealer,Manual,Test Drive Car,1780,5,High,Low,0.0,Petrol_Manual,Ford,803.96,0
1780,CAR_001781,Ford Ecosport Thunder Edition Diesel,2020,1331000.0,1010.0,Diesel,Dealer,Manual,Test Drive Car,1781,5,Low,Low,0.0,Diesel_Manual,Ford,1317.82,1
1781,CAR_001782,Ford Freestyle Titanium Plus,2020,852000.0,1010.0,LPG,Dealer,Manual,Test Drive Car,1782,5,High,Low,0.0,Petrol_Manual,Ford,843.56,0
1782,CAR_001783,Tata Indica DLE,2006,45000.0,120000.0,Diesel,Individual,Manual,Second Owner,1783,19,Low,High,0.0,Diesel_Manual,Tata,0.38,1
1783,CAR_001784,Renault Duster 85PS Diesel RxL Optional,2013,430000.0,70000.0,Diesel,Individual,Manual,Third Owner,1784,12,Mid,Medium,0.0,Diesel_Manual,Renault,6.14,1
1784,CAR_001785,Tata Indigo LX,2011,140000.0,60000.0,Electric,Individual,Manual,Second Owner,1785,14,Low,Medium,0.0,Diesel_Manual,Tata,2.33,1
1785,CAR_001786,Mahindra Scorpio S2 9 Seater,2015,830000.0,60000.0,Diesel,Individual,Manual,First Owner,1786,10,High,Medium,0.0,Diesel_Manual,Mahindra,11.86,1
1786,CAR_001787,Mahindra Scorpio VLS 2.2 mHawk,2009,409999.0,110000.0,Petrol,Individual,Manual,Second Owner,1787,16,Mid,Medium,0.0,Diesel_Manual,Mahindra,3.73,1
1787,CAR_001788,Maruti Swift VDI,2012,190000.0,60000.0,Diesel,Individual,Manual,Third Owner,1788,13,Low,High,0.0,Diesel_Manual,Maruti,2.38,1
1788,CAR_001789,Honda Accord VTi-L (MT),2007,200000.0,80000.0,Petrol,Individual,Manual,Second Owner,1789,18,Low,Medium,0.0,Petrol_Manual,Honda,2.5,0
1789,CAR_001790,Hyundai Accent GLX,2006,120000.0,110000.0,Petrol,Individual,Manual,Second Owner,1790,19,Low,High,0.0,Petrol_Manual,Hyundai,1.09,0
1790,CAR_001791,Mahindra Bolero Power Plus Plus AC BSIV PS,2019,850000.0,50000.0,Diesel,Individual,Manual,First Owner,1791,6,High,Medium,0.0,Diesel_Manual,Mahindra,17.0,1
1791,CAR_001792,Tata Indigo CS LE (TDI) BS-III,2011,95000.0,115000.0,Diesel,Individual,Manual,Fourth & Above Owner,1792,14,Low,High,0.0,Diesel_Manual,Tata,0.83,1
1792,CAR_001793,Maruti Wagon R LXI BS IV,2013,213000.0,80000.0,Petrol,Individual,Manual,First Owner,1793,12,Low,High,0.0,Petrol_Manual,Maruti,2.66,0
1793,CAR_001794,Volkswagen Ameo 1.5 TDI Highline Plus 16,2018,660000.0,25000.0,Diesel,Individual,Manual,First Owner,1794,7,High,Low,0.0,Diesel_Manual,Volkswagen,26.4,1
1794,CAR_001795,Nissan Micra XL,2017,415000.0,60000.0,Petrol,Dealer,Manual,First Owner,1795,8,Mid,Medium,0.0,Petrol_Manual,Nissan,8.48,0
1795,CAR_001796,Ford Ecosport 1.5 Petrol Titanium Plus,2019,1100000.0,5166.0,Petrol,Dealer,Manual,Test Drive Car,1796,6,Premium,Low,0.0,Petrol_Manual,Ford,212.93,0
1796,CAR_001797,Mahindra XUV500 W8 4WD,2012,650000.0,76290.0,Diesel,Dealer,Manual,First Owner,1797,13,High,Medium,0.0,Diesel_Manual,Mahindra,8.52,1
1797,CAR_001798,Mahindra XUV500 AT W10 AWD,2016,969999.0,70000.0,Diesel,Individual,Automatic,Second Owner,1798,9,High,Medium,0.0,Diesel_Automatic,Mahindra,13.86,1
1798,CAR_001799,Mahindra XUV500 W10 2WD,2016,1250000.0,35000.0,Diesel,Individual,Manual,First Owner,1799,9,Premium,Medium,0.0,Diesel_Manual,Mahindra,35.71,1
1799,CAR_001800,Maruti Swift Dzire AMT VDI,2017,715000.0,25000.0,Diesel,Individual,Automatic,First Owner,1800,8,High,Low,0.0,Diesel_Automatic,Maruti,28.6,1
1800,CAR_001801,Maruti Swift Dzire AMT VDI,2017,715000.0,25000.0,Diesel,Individual,Automatic,First Owner,1801,8,Low,Low,0.0,Diesel_Automatic,Maruti,28.6,1
1801,CAR_001802,Maruti Swift VDI,2016,565000.0,47000.0,Diesel,Dealer,Manual,First Owner,1802,9,Mid,Medium,0.0,Diesel_Manual,Maruti,12.02,1
1802,CAR_001803,Mahindra Thar 4X4,2013,615000.0,45766.0,Diesel,Individual,Manual,Second Owner,1803,12,High,Medium,0.0,Diesel_Manual,Mahindra,13.44,1
1803,CAR_001804,Tata New Safari 4X4,2007,199000.0,78771.0,Diesel,Dealer,Manual,Second Owner,1804,18,Low,High,0.0,Diesel_Manual,Tata,2.53,1
1804,CAR_001805,Maruti Swift Vdi BSIII,2010,325000.0,60000.0,Diesel,Dealer,Manual,First Owner,1805,15,Mid,High,0.0,Diesel_Manual,Maruti,4.1,1
1805,CAR_001806,Ford Figo Diesel EXI,2012,409999.0,28000.0,Diesel,Individual,Manual,First Owner,1806,13,Mid,Low,0.0,Diesel_Manual,Ford,14.64,1
1806,CAR_001807,Hyundai Verna SX CRDi AT,2012,500000.0,72000.0,Diesel,Dealer,Automatic,Second Owner,1807,13,Mid,High,0.0,Diesel_Automatic,Hyundai,6.94,1
1807,CAR_001808,Ford Figo Diesel LXI,2011,225000.0,60000.0,Diesel,Dealer,Manual,Second Owner,1808,14,Low,High,0.0,Diesel_Manual,Ford,2.93,1
1808,CAR_001809,Maruti Alto LX BSIII,2007,135000.0,70000.0,CNG,Individual,Manual,Second Owner,1809,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.93,0
1809,CAR_001810,Renault Koleos 2.0 Diesel,2012,750000.0,80000.0,Diesel,Individual,Manual,Second Owner,1810,13,Low,High,0.0,Diesel_Automatic,Renault,9.38,1
1810,CAR_001811,Maruti Wagon R LXI DUO BSIII,2009,155000.0,60000.0,LPG,Individual,Manual,Second Owner,1811,16,Low,Medium,0.0,LPG_Manual,Maruti,2.58,0
1811,CAR_001812,Volkswagen Polo Diesel Highline 1.2L,2012,335000.0,77000.0,Diesel,Dealer,Manual,Third Owner,1812,13,Mid,High,0.0,Diesel_Manual,Volkswagen,4.35,1
1812,CAR_001813,Ford Ikon 1.8 D,2005,114999.0,92645.0,Diesel,Dealer,Manual,Fourth & Above Owner,1813,20,Low,High,0.0,Diesel_Manual,Ford,1.24,1
1813,CAR_001814,Mercedes-Benz New C-Class 200 CDI Classic,2007,699000.0,101849.0,Diesel,Dealer,Manual,Second Owner,1814,18,High,High,0.0,Diesel_Manual,Mercedes-Benz,6.86,1
1814,CAR_001815,Mahindra Bolero DI,2004,110000.0,120000.0,Diesel,Individual,Manual,Fourth & Above Owner,1815,21,Low,High,0.0,Diesel_Manual,Mahindra,0.92,1
1815,CAR_001816,Hyundai i20 Asta 1.2,2012,300000.0,80000.0,Petrol,Individual,Manual,First Owner,1816,13,Low,High,0.0,Petrol_Manual,Hyundai,3.75,0
1816,CAR_001817,Mahindra Scorpio S4 4WD,2014,700000.0,70000.0,Diesel,Individual,Manual,Second Owner,1817,11,High,Medium,0.0,Diesel_Manual,Mahindra,10.0,1
1817,CAR_001818,Maruti Ertiga SHVS LDI Option,2017,700000.0,120000.0,Diesel,Individual,Manual,First Owner,1818,8,High,High,0.0,Diesel_Manual,Maruti,5.83,1
1818,CAR_001819,Maruti Alto 800 LXI,2013,210000.0,42000.0,Petrol,Individual,Manual,Third Owner,1819,12,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
1819,CAR_001820,Maruti Ertiga VDI,2012,600000.0,120000.0,Diesel,Individual,Manual,Third Owner,1820,13,Mid,High,0.0,Diesel_Manual,Maruti,5.0,1
1820,CAR_001821,Tata Manza Aura (ABS) Quadrajet BS IV,2012,180000.0,155836.0,Diesel,Individual,Manual,Second Owner,1821,13,Low,Very High,0.0,Diesel_Manual,Tata,1.16,1
1821,CAR_001822,Maruti Wagon R VXi BSII,2011,220000.0,59000.0,Petrol,Individual,Manual,Second Owner,1822,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.73,0
1822,CAR_001823,Chevrolet Spark 1.0 LT,2009,70000.0,60000.0,Petrol,Individual,Manual,First Owner,1823,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.17,0
1823,CAR_001824,Hyundai Creta 1.4 CRDi Base,2018,950000.0,70000.0,Diesel,Individual,Manual,Second Owner,1824,7,Low,Medium,0.0,Diesel_Manual,Hyundai,13.57,1
1824,CAR_001825,Ford Fiesta 1.5 TDCi Ambiente,2014,800000.0,60000.0,Diesel,Individual,Manual,First Owner,1825,11,High,Low,0.0,Diesel_Manual,Ford,40.0,1
1825,CAR_001826,Mahindra XUV500 W8 4WD,2012,700000.0,120000.0,Diesel,Individual,Manual,Third Owner,1826,13,High,High,0.0,Diesel_Manual,Mahindra,5.83,1
1826,CAR_001827,Renault KWID 1.0,2016,265000.0,40000.0,Petrol,Individual,Manual,First Owner,1827,9,Low,Medium,0.0,Petrol_Manual,Renault,6.62,0
1827,CAR_001828,Volkswagen Polo Diesel Trendline 1.2L,2011,175000.0,90000.0,Diesel,Individual,Manual,Third Owner,1828,14,Low,High,0.0,Diesel_Manual,Volkswagen,1.94,1
1828,CAR_001829,Mahindra Quanto C6,2012,260000.0,89000.0,Diesel,Individual,Manual,Second Owner,1829,13,Low,High,0.0,Diesel_Manual,Mahindra,2.92,1
1829,CAR_001830,Toyota Innova 2.5 VX (Diesel) 7 Seater,2013,950000.0,60000.0,Diesel,Individual,Manual,Third Owner,1830,12,High,High,0.0,Diesel_Manual,Toyota,11.88,1
1830,CAR_001831,Maruti Swift Vdi BSIII,2009,250000.0,200000.0,Diesel,Individual,Manual,Fourth & Above Owner,1831,16,Low,Very High,0.0,Diesel_Manual,Maruti,1.25,1
1831,CAR_001832,Mahindra Scorpio S10 7 Seater,2015,4461000.0,50000.0,Diesel,Individual,Manual,First Owner,1832,10,High,Medium,0.0,Diesel_Manual,Mahindra,12.6,1
1832,CAR_001833,Tata Nano LX SE,2012,35000.0,35000.0,Petrol,Individual,Manual,Third Owner,1833,13,Low,Medium,0.0,Petrol_Manual,Tata,1.0,0
1833,CAR_001834,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,1834,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
1834,CAR_001835,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,1835,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
1835,CAR_001836,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,1836,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
1836,CAR_001837,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,1837,6,Low,Low,0.0,Diesel_Automatic,BMW,165.0,1
1837,CAR_001838,Hyundai Creta 1.6 CRDi SX,2016,4461000.0,52600.0,LPG,Individual,Manual,First Owner,1838,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
1838,CAR_001839,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,1839,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
1839,CAR_001840,Hyundai Grand i10 1.2 Kappa Magna AT,2017,4461000.0,60000.0,Electric,Dealer,Automatic,First Owner,1840,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
1840,CAR_001841,Maruti Swift ZXI BSIV,2016,4461000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,1841,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
1841,CAR_001842,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,1842,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
1842,CAR_001843,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,1843,8,Low,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
1843,CAR_001844,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,1844,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
1844,CAR_001845,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Petrol,Trustmark Dealer,Automatic,First Owner,1845,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
1845,CAR_001846,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,1846,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
1846,CAR_001847,Hyundai i10 Magna,2012,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,1847,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
1847,CAR_001848,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,1848,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
1848,CAR_001849,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,1849,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
1849,CAR_001850,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,1850,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
1850,CAR_001851,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,1851,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
1851,CAR_001852,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,1852,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
1852,CAR_001853,Chevrolet Enjoy TCDi LS 8 Seater,2014,330000.0,200000.0,CNG,Individual,Manual,Third Owner,1853,11,Mid,Very High,0.0,Diesel_Manual,Chevrolet,1.65,1
1853,CAR_001854,Maruti Alto LXi,2009,150000.0,120000.0,Petrol,Individual,Manual,First Owner,1854,16,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
1854,CAR_001855,Tata Indigo LS,2008,105000.0,90000.0,Diesel,Individual,Manual,First Owner,1855,17,Low,High,0.0,Diesel_Manual,Tata,1.17,1
1855,CAR_001856,Tata Indica Vista TDI LX,2013,185000.0,65000.0,Diesel,Individual,Manual,First Owner,1856,12,Low,Medium,0.0,Diesel_Manual,Tata,2.85,1
1856,CAR_001857,Fiat Grande Punto EVO 1.3 Active,2014,260000.0,50000.0,Diesel,Individual,Manual,First Owner,1857,11,Low,Medium,0.0,Diesel_Manual,Fiat,5.2,1
1857,CAR_001858,Maruti Swift Dzire VDi,2012,450000.0,33000.0,Diesel,Individual,Manual,First Owner,1858,13,Mid,Medium,0.0,Diesel_Manual,Maruti,13.64,1
1858,CAR_001859,Maruti Swift VXI Deca,2016,500000.0,38000.0,Petrol,Individual,Manual,First Owner,1859,9,Mid,Medium,0.0,Petrol_Manual,Maruti,13.16,0
1859,CAR_001860,Maruti Alto K10 2010-2014 VXI,2012,260000.0,40000.0,Petrol,Individual,Manual,First Owner,1860,13,Low,Medium,0.0,Petrol_Manual,Maruti,6.5,0
1860,CAR_001861,Volkswagen Polo Diesel Trendline 1.2L,2012,265000.0,55000.0,Diesel,Individual,Manual,Second Owner,1861,13,Low,Medium,0.0,Diesel_Manual,Volkswagen,4.82,1
1861,CAR_001862,Mahindra Thar CRDe,2017,800000.0,26000.0,Diesel,Individual,Manual,Second Owner,1862,8,High,Medium,0.0,Diesel_Manual,Mahindra,30.77,1
1862,CAR_001863,Tata Safari Storme VX Varicor 400,2016,869999.0,80000.0,Diesel,Individual,Manual,First Owner,1863,9,High,Medium,0.0,Diesel_Manual,Tata,10.87,1
1863,CAR_001864,Maruti Baleno Alpha 1.2,2018,690000.0,40000.0,Petrol,Individual,Manual,First Owner,1864,7,High,Medium,0.0,Petrol_Manual,Maruti,17.25,0
1864,CAR_001865,Mitsubishi Pajero 2.8 SFX BSIV Dual Tone,2011,825000.0,60000.0,Diesel,Individual,Manual,First Owner,1865,14,High,High,0.0,Diesel_Manual,Mitsubishi,6.88,1
1865,CAR_001866,Maruti Alto LX BSIII,2006,160000.0,63230.0,Petrol,Individual,Manual,Second Owner,1866,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.53,0
1866,CAR_001867,Mahindra Scorpio 2.6 SLX CRDe,2006,300000.0,120000.0,Diesel,Individual,Manual,Second Owner,1867,19,Low,High,0.0,Diesel_Manual,Mahindra,2.5,1
1867,CAR_001868,Maruti Wagon R VXI,2005,100000.0,100000.0,Petrol,Individual,Manual,Third Owner,1868,20,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
1868,CAR_001869,Hyundai i20 Asta 1.4 CRDi,2015,530000.0,77000.0,Diesel,Individual,Manual,Second Owner,1869,10,Mid,High,0.0,Diesel_Manual,Hyundai,6.88,1
1869,CAR_001870,Maruti Swift VXI Optional,2016,360000.0,30000.0,Petrol,Individual,Manual,First Owner,1870,9,Mid,Low,0.0,Petrol_Manual,Maruti,12.0,0
1870,CAR_001871,Maruti Swift 1.2 DLX,2015,220000.0,40000.0,Petrol,Individual,Manual,First Owner,1871,10,Low,Medium,0.0,Petrol_Manual,Maruti,5.5,0
1871,CAR_001872,Maruti Alto 800 LXI,2017,178000.0,35000.0,Petrol,Individual,Manual,First Owner,1872,8,Low,Medium,0.0,Petrol_Manual,Maruti,5.09,0
1872,CAR_001873,Mahindra Thar DI 4X2,2018,515000.0,5000.0,Diesel,Individual,Manual,First Owner,1873,7,Mid,Low,0.0,Diesel_Manual,Mahindra,103.0,1
1873,CAR_001874,Maruti Ritz VXi,2014,300000.0,35000.0,Petrol,Individual,Manual,First Owner,1874,11,Low,Medium,0.0,Petrol_Manual,Maruti,8.57,0
1874,CAR_001875,Tata Indigo LS,2008,105000.0,90000.0,Diesel,Individual,Manual,First Owner,1875,17,Low,Medium,0.0,Diesel_Manual,Tata,1.17,1
1875,CAR_001876,Tata Indica Vista TDI LX,2013,185000.0,65000.0,Diesel,Individual,Manual,First Owner,1876,12,Low,Medium,0.0,Diesel_Manual,Tata,2.85,1
1876,CAR_001877,Tata Indica Vista Quadrajet LS,2012,150000.0,80000.0,Diesel,Individual,Manual,Second Owner,1877,13,Low,High,0.0,Diesel_Manual,Tata,1.88,1
1877,CAR_001878,Fiat Grande Punto EVO 1.3 Active,2014,260000.0,50000.0,Diesel,Individual,Manual,First Owner,1878,11,Low,Medium,0.0,Diesel_Manual,Fiat,5.2,1
1878,CAR_001879,Maruti Swift Dzire VDi,2012,450000.0,33000.0,Diesel,Individual,Manual,First Owner,1879,13,Mid,Medium,0.0,Diesel_Manual,Maruti,13.64,1
1879,CAR_001880,Maruti Swift VXI Deca,2016,500000.0,38000.0,Petrol,Individual,Manual,First Owner,1880,9,Mid,Medium,0.0,Petrol_Manual,Maruti,13.16,0
1880,CAR_001881,Maruti Alto K10 2010-2014 VXI,2012,260000.0,40000.0,Petrol,Individual,Manual,First Owner,1881,13,Low,Medium,0.0,Petrol_Manual,Maruti,6.5,0
1881,CAR_001882,Volkswagen Polo Diesel Trendline 1.2L,2012,265000.0,55000.0,Diesel,Individual,Manual,Second Owner,1882,13,Low,Medium,0.0,Diesel_Manual,Volkswagen,4.82,1
1882,CAR_001883,Tata Nexon 1.2 Revotron XZ Plus Dual Tone,2017,750000.0,15000.0,Petrol,Individual,Manual,First Owner,1883,8,High,Low,0.0,Petrol_Manual,Tata,50.0,0
1883,CAR_001884,Mahindra Thar CRDe,2017,800000.0,26000.0,Diesel,Individual,Manual,Second Owner,1884,8,High,Low,0.0,Diesel_Manual,Mahindra,30.77,1
1884,CAR_001885,Tata Safari Storme VX Varicor 400,2016,869999.0,80000.0,Diesel,Individual,Manual,First Owner,1885,9,High,High,0.0,Diesel_Manual,Tata,10.87,1
1885,CAR_001886,Toyota Innova 2.5 G1 BSIV,2012,950000.0,60000.0,Diesel,Individual,Manual,First Owner,1886,13,High,High,0.0,Diesel_Manual,Toyota,11.88,1
1886,CAR_001887,Maruti Baleno Alpha 1.2,2018,690000.0,40000.0,Petrol,Individual,Manual,First Owner,1887,7,High,Medium,0.0,Petrol_Manual,Maruti,17.25,0
1887,CAR_001888,Mitsubishi Pajero 2.8 SFX BSIV Dual Tone,2011,825000.0,120000.0,Diesel,Individual,Manual,First Owner,1888,14,High,High,0.0,Diesel_Manual,Mitsubishi,6.88,1
1888,CAR_001889,Maruti 800 Std,2013,170000.0,70000.0,Petrol,Individual,Manual,First Owner,1889,12,Low,Medium,0.0,Petrol_Manual,Maruti,2.43,0
1889,CAR_001890,Maruti Alto LX BSIII,2006,160000.0,63230.0,Petrol,Individual,Manual,Second Owner,1890,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.53,0
1890,CAR_001891,Mahindra Scorpio 2.6 SLX CRDe,2006,300000.0,120000.0,Diesel,Individual,Manual,Second Owner,1891,19,Low,High,0.0,Diesel_Manual,Mahindra,2.5,1
1891,CAR_001892,Datsun GO Plus A,2015,315000.0,45000.0,Petrol,Individual,Manual,Second Owner,1892,10,Mid,Medium,0.0,Petrol_Manual,Datsun,7.0,0
1892,CAR_001893,Maruti 800 Std BSII,2006,90000.0,40000.0,Petrol,Individual,Manual,Second Owner,1893,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.25,0
1893,CAR_001894,Maruti Wagon R VXI,2005,100000.0,100000.0,Petrol,Individual,Manual,Third Owner,1894,20,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
1894,CAR_001895,Tata Indica GLS BS IV,2008,65000.0,110000.0,Petrol,Individual,Manual,Third Owner,1895,17,Low,High,0.0,Petrol_Manual,Tata,0.59,0
1895,CAR_001896,Hyundai Verna 1.6 CRDI,2012,312000.0,90000.0,Diesel,Individual,Manual,Second Owner,1896,13,Mid,Medium,0.0,Diesel_Manual,Hyundai,3.47,1
1896,CAR_001897,Maruti Vitara Brezza LDi,2018,700000.0,25000.0,Diesel,Individual,Manual,First Owner,1897,7,High,Low,0.0,Diesel_Manual,Maruti,28.0,1
1897,CAR_001898,Hyundai Santro Xing XL eRLX Euro III,2005,150000.0,47000.0,Petrol,Individual,Manual,First Owner,1898,20,Low,Medium,0.0,Petrol_Manual,Hyundai,3.19,0
1898,CAR_001899,Renault Fluence 1.5,2013,450000.0,60000.0,Diesel,Individual,Manual,First Owner,1899,12,Mid,Medium,0.0,Diesel_Manual,Renault,7.5,1
1899,CAR_001900,Hyundai Verna Transform CRDi VGT ABS,2010,229999.0,120000.0,LPG,Individual,Manual,Second Owner,1900,15,Low,Medium,0.0,Diesel_Manual,Hyundai,1.92,1
1900,CAR_001901,Maruti Swift Ldi BSIII,2008,111000.0,120000.0,Diesel,Individual,Manual,First Owner,1901,17,Low,High,0.0,Diesel_Manual,Maruti,0.92,1
1901,CAR_001902,Maruti Swift Dzire VDi,2012,250000.0,140000.0,Diesel,Individual,Manual,Second Owner,1902,13,Low,High,0.0,Diesel_Manual,Maruti,1.79,1
1902,CAR_001903,Ford Freestyle Titanium Plus Diesel,2019,750000.0,1001.0,Electric,Dealer,Manual,First Owner,1903,6,High,Low,0.0,Diesel_Manual,Ford,749.25,1
1903,CAR_001904,Ford Figo 1.5P Titanium AT,2017,774000.0,1758.0,Petrol,Dealer,Automatic,First Owner,1904,8,High,Low,0.0,Petrol_Automatic,Ford,440.27,0
1904,CAR_001905,Ford Figo 1.5P Titanium AT,2018,711000.0,1452.0,Petrol,Dealer,Automatic,First Owner,1905,7,Low,Low,0.0,Petrol_Automatic,Ford,489.67,0
1905,CAR_001906,Ford Figo 1.5D Titanium MT,2017,600000.0,35122.0,Diesel,Dealer,Manual,First Owner,1906,8,Mid,Medium,0.0,Diesel_Manual,Ford,17.08,1
1906,CAR_001907,Maruti Swift Dzire VDI,2015,455000.0,92621.0,Diesel,Dealer,Manual,First Owner,1907,10,Mid,High,0.0,Diesel_Manual,Maruti,4.91,1
1907,CAR_001908,Datsun GO T Petrol,2015,225000.0,92198.0,Petrol,Dealer,Manual,Second Owner,1908,10,Low,High,0.0,Petrol_Manual,Datsun,2.44,0
1908,CAR_001909,Maruti Swift Dzire VDI,2019,650000.0,5000.0,Petrol,Individual,Manual,First Owner,1909,6,Low,Medium,0.0,Diesel_Manual,Maruti,130.0,1
1909,CAR_001910,Chevrolet Beat LT,2013,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,1910,12,Low,Medium,0.0,Petrol_Manual,Chevrolet,4.0,0
1910,CAR_001911,Hyundai Creta 1.6 CRDi AT SX Plus,2016,875000.0,74000.0,Diesel,Individual,Automatic,First Owner,1911,9,High,High,0.0,Diesel_Automatic,Hyundai,11.82,1
1911,CAR_001912,Skoda Rapid 1.5 TDI AT Ambition BSIV,2017,750000.0,40000.0,Diesel,Individual,Automatic,First Owner,1912,8,High,Medium,0.0,Diesel_Automatic,Skoda,18.75,1
1912,CAR_001913,Maruti Swift Dzire AMT VDI,2019,725000.0,22000.0,Diesel,Individual,Automatic,First Owner,1913,6,High,Low,0.0,Diesel_Automatic,Maruti,32.95,1
1913,CAR_001914,Maruti Alto LXi,2007,100000.0,100000.0,Petrol,Individual,Manual,Second Owner,1914,18,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
1914,CAR_001915,Tata Hexa XT,2019,1100000.0,40000.0,Diesel,Individual,Manual,First Owner,1915,6,Low,Medium,0.0,Diesel_Manual,Tata,27.5,1
1915,CAR_001916,Ford Endeavour 3.2 Titanium AT 4X4,2017,4461000.0,30000.0,Diesel,Individual,Automatic,First Owner,1916,8,Premium,Low,0.0,Diesel_Automatic,Ford,90.0,1
1916,CAR_001917,Renault KWID RXT Optional,2016,280000.0,23000.0,Diesel,Individual,Manual,First Owner,1917,9,Low,Low,0.0,Petrol_Manual,Renault,12.17,0
1917,CAR_001918,Tata Indica Vista Aqua TDI BSIII,2009,150000.0,60000.0,Diesel,Individual,Manual,First Owner,1918,16,Low,Low,0.0,Diesel_Manual,Tata,5.0,1
1918,CAR_001919,Tata Manza Aura Safire,2010,200000.0,50000.0,Petrol,Individual,Manual,First Owner,1919,15,Low,Medium,0.0,Petrol_Manual,Tata,4.0,0
1919,CAR_001920,Maruti Ritz VDi,2010,148000.0,100000.0,Diesel,Individual,Manual,Second Owner,1920,15,Low,High,0.0,Diesel_Manual,Maruti,1.48,1
1920,CAR_001921,Maruti Swift Dzire VXi,2011,254999.0,70000.0,Petrol,Individual,Manual,Second Owner,1921,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.64,0
1921,CAR_001922,Maruti Swift Dzire ZXI,2013,360000.0,60000.0,Petrol,Individual,Manual,Second Owner,1922,12,Low,Medium,0.0,Petrol_Manual,Maruti,6.0,0
1922,CAR_001923,Maruti Alto LX,2006,95000.0,90000.0,CNG,Individual,Manual,Second Owner,1923,19,Low,High,0.0,Petrol_Manual,Maruti,1.06,0
1923,CAR_001924,Mahindra Bolero SLE BSIII,2007,185000.0,230000.0,Diesel,Individual,Manual,Second Owner,1924,18,Low,Very High,0.0,Diesel_Manual,Mahindra,0.8,1
1924,CAR_001925,Tata Indica Vista Quadrajet VX,2012,185000.0,40000.0,Diesel,Individual,Manual,First Owner,1925,13,Low,Medium,0.0,Diesel_Manual,Tata,4.62,1
1925,CAR_001926,Renault KWID RXL,2019,300000.0,15000.0,Petrol,Individual,Manual,First Owner,1926,6,Low,Low,0.0,Petrol_Manual,Renault,20.0,0
1926,CAR_001927,Hyundai Verna VTVT 1.6 SX,2018,800000.0,40000.0,Petrol,Individual,Manual,First Owner,1927,7,High,Medium,0.0,Petrol_Manual,Hyundai,20.0,0
1927,CAR_001928,Maruti Zen LX,2000,60000.0,50000.0,Petrol,Individual,Manual,First Owner,1928,25,Low,Medium,0.0,Petrol_Manual,Maruti,1.2,0
1928,CAR_001929,Fiat Punto 1.2 Active,2010,80000.0,50000.0,Petrol,Individual,Manual,Second Owner,1929,15,Low,Medium,0.0,Petrol_Manual,Fiat,1.6,0
1929,CAR_001930,Hyundai Xcent 1.2 VTVT S,2018,500000.0,8000.0,Petrol,Individual,Manual,First Owner,1930,7,Mid,Low,0.0,Petrol_Manual,Hyundai,62.5,0
1930,CAR_001931,Hyundai EON Era Plus,2013,250000.0,35000.0,Petrol,Individual,Manual,First Owner,1931,12,Low,Medium,0.0,Petrol_Manual,Hyundai,7.14,0
1931,CAR_001932,Tata New Safari DICOR 2.2 EX 4x2,2010,250000.0,80000.0,Diesel,Individual,Manual,First Owner,1932,15,Low,High,0.0,Diesel_Manual,Tata,3.12,1
1932,CAR_001933,Maruti Wagon R LXI BSIII,2005,125000.0,50000.0,Petrol,Individual,Manual,Second Owner,1933,20,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
1933,CAR_001934,Maruti Alto LXi BSII,2002,100000.0,80000.0,Petrol,Individual,Manual,Fourth & Above Owner,1934,23,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
1934,CAR_001935,Mahindra Scorpio M2DI,2010,480000.0,90000.0,Diesel,Individual,Manual,First Owner,1935,15,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.33,1
1935,CAR_001936,Mahindra KUV 100 G80 K2,2018,320000.0,30000.0,Petrol,Individual,Manual,Second Owner,1936,7,Mid,Low,0.0,Petrol_Manual,Mahindra,10.67,0
1936,CAR_001937,Maruti A-Star Vxi,2010,130000.0,70000.0,Petrol,Individual,Manual,Second Owner,1937,15,Low,Medium,0.0,Petrol_Manual,Maruti,1.86,0
1937,CAR_001938,Tata Indica DLS,2006,4461000.0,80000.0,Diesel,Individual,Manual,Second Owner,1938,19,Low,High,0.0,Diesel_Manual,Tata,0.71,1
1938,CAR_001939,Hyundai Santro GLS I - Euro I,2005,75000.0,60000.0,Petrol,Individual,Manual,First Owner,1939,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.25,0
1939,CAR_001940,Maruti 800 DX,2003,60000.0,35000.0,Petrol,Individual,Manual,First Owner,1940,22,Low,Medium,0.0,Petrol_Manual,Maruti,1.71,0
1940,CAR_001941,Maruti Alto LXi BSIII,2008,125000.0,70000.0,Petrol,Individual,Manual,First Owner,1941,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.79,0
1941,CAR_001942,Honda City 1.5 EXI S,2004,105000.0,60000.0,Petrol,Individual,Manual,Third Owner,1942,21,Low,High,0.0,Petrol_Manual,Honda,1.05,0
1942,CAR_001943,Maruti Wagon R LXI DUO BSIII,2008,125000.0,152000.0,LPG,Individual,Manual,Second Owner,1943,17,Low,Very High,0.0,LPG_Manual,Maruti,0.82,0
1943,CAR_001944,Maruti Zen LXI,1998,69000.0,70000.0,Petrol,Individual,Manual,Second Owner,1944,27,Low,Medium,0.0,Petrol_Manual,Maruti,0.99,0
1944,CAR_001945,Tata Indigo LX,2012,130000.0,90000.0,Diesel,Individual,Manual,Second Owner,1945,13,Low,High,0.0,Diesel_Manual,Tata,1.44,1
1945,CAR_001946,Maruti Baleno Zeta 1.2,2019,630000.0,60000.0,Petrol,Individual,Manual,First Owner,1946,6,High,Medium,0.0,Petrol_Manual,Maruti,15.75,0
1946,CAR_001947,Honda Amaze E i-VTEC,2016,330000.0,50000.0,Petrol,Individual,Manual,Second Owner,1947,9,Mid,Medium,0.0,Petrol_Manual,Honda,6.6,0
1947,CAR_001948,Maruti Wagon R VXI BS IV,2014,340000.0,15000.0,Petrol,Individual,Manual,Second Owner,1948,11,Mid,Low,0.0,Petrol_Manual,Maruti,22.67,0
1948,CAR_001949,Hyundai Santro Xing XG eRLX Euro III,2006,60000.0,120000.0,Petrol,Individual,Manual,Third Owner,1949,19,Low,High,0.0,Petrol_Manual,Hyundai,0.5,0
1949,CAR_001950,Tata Safari Storme EX,2014,550000.0,78322.0,Diesel,Individual,Manual,First Owner,1950,11,Mid,High,0.0,Diesel_Manual,Tata,7.02,1
1950,CAR_001951,Maruti Ertiga VDI,2014,480000.0,60000.0,Diesel,Individual,Manual,Second Owner,1951,11,Low,High,0.0,Diesel_Manual,Maruti,4.36,1
1951,CAR_001952,Hyundai Verna 1.6 SX CRDi (O),2013,315000.0,80000.0,Diesel,Individual,Manual,Second Owner,1952,12,Mid,High,0.0,Diesel_Manual,Hyundai,3.94,1
1952,CAR_001953,Ford Fiesta Titanium 1.5 TDCi,2012,165000.0,60000.0,Diesel,Individual,Manual,First Owner,1953,13,Low,High,0.0,Diesel_Manual,Ford,1.38,1
1953,CAR_001954,Chevrolet Beat LS,2016,170000.0,125000.0,Petrol,Individual,Manual,First Owner,1954,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.36,0
1954,CAR_001955,Maruti Baleno Delta 1.2,2018,550000.0,8000.0,Petrol,Individual,Manual,First Owner,1955,7,Mid,Low,0.0,Petrol_Manual,Maruti,68.75,0
1955,CAR_001956,Volkswagen Ameo 1.2 MPI Trendline,2018,465000.0,13000.0,Petrol,Individual,Manual,First Owner,1956,7,Mid,Low,0.0,Petrol_Manual,Volkswagen,35.77,0
1956,CAR_001957,Tata Indica Vista Quadrajet LX,2012,120000.0,80000.0,Diesel,Individual,Manual,Third Owner,1957,13,Low,High,0.0,Diesel_Manual,Tata,1.5,1
1957,CAR_001958,Hyundai Verna 1.6 Xi ABS,2008,220000.0,54309.0,Petrol,Individual,Manual,First Owner,1958,17,Low,Medium,0.0,Petrol_Manual,Hyundai,4.05,0
1958,CAR_001959,Ford Figo Diesel LXI,2011,180000.0,70000.0,Diesel,Individual,Manual,Second Owner,1959,14,Low,Medium,0.0,Diesel_Manual,Ford,2.57,1
1959,CAR_001960,Chevrolet Aveo 1.6 LT,2006,150000.0,34600.0,LPG,Individual,Manual,Third Owner,1960,19,Low,Medium,0.0,Petrol_Manual,Chevrolet,4.34,0
1960,CAR_001961,Tata Zest Revotron 1.2T XMS,2016,520000.0,25000.0,Petrol,Individual,Manual,First Owner,1961,9,Mid,Low,0.0,Petrol_Manual,Tata,20.8,0
1961,CAR_001962,Tata Safari Storme EX,2013,430000.0,80000.0,Electric,Individual,Manual,Third Owner,1962,12,Mid,High,0.0,Diesel_Manual,Tata,5.38,1
1962,CAR_001963,Mahindra Bolero DI DX 8 Seater,2009,300000.0,60000.0,Diesel,Individual,Manual,Second Owner,1963,16,Low,Medium,0.0,Diesel_Manual,Mahindra,5.45,1
1963,CAR_001964,Hyundai Venue SX Opt Diesel,2020,1000000.0,5000.0,Diesel,Individual,Manual,First Owner,1964,5,High,Medium,0.0,Diesel_Manual,Hyundai,200.0,1
1964,CAR_001965,Honda WR-V i-DTEC VX,2017,725000.0,30000.0,Petrol,Individual,Manual,First Owner,1965,8,High,Low,0.0,Diesel_Manual,Honda,24.17,1
1965,CAR_001966,Hyundai Verna 1.4 EX,2011,500000.0,31000.0,Diesel,Individual,Manual,First Owner,1966,14,Mid,Medium,0.0,Diesel_Manual,Hyundai,16.13,1
1966,CAR_001967,Maruti Wagon R LXI BS IV,2011,250000.0,50000.0,Petrol,Individual,Manual,First Owner,1967,14,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
1967,CAR_001968,Maruti Swift Dzire VDI,2018,700000.0,38217.0,Diesel,Individual,Manual,First Owner,1968,7,High,Medium,0.0,Diesel_Manual,Maruti,18.32,1
1968,CAR_001969,Ford Figo Petrol ZXI,2011,185000.0,50000.0,Petrol,Individual,Manual,Third Owner,1969,14,Low,Medium,0.0,Petrol_Manual,Ford,3.7,0
1969,CAR_001970,Maruti Ritz VDi,2011,200000.0,120000.0,Diesel,Individual,Manual,First Owner,1970,14,Low,High,0.0,Diesel_Manual,Maruti,1.67,1
1970,CAR_001971,Mahindra Bolero DI,2004,315000.0,110000.0,Diesel,Individual,Manual,Third Owner,1971,21,Mid,High,0.0,Diesel_Manual,Mahindra,2.86,1
1971,CAR_001972,Tata Hexa XT,2019,1100000.0,40000.0,Diesel,Individual,Manual,First Owner,1972,6,Premium,Medium,0.0,Diesel_Manual,Tata,27.5,1
1972,CAR_001973,Hyundai Santro LP zipPlus,2001,52000.0,50000.0,Petrol,Individual,Manual,Third Owner,1973,24,Low,Medium,0.0,Petrol_Manual,Hyundai,1.04,0
1973,CAR_001974,Hyundai i20 Active 1.2 S,2017,600000.0,10000.0,Petrol,Individual,Manual,First Owner,1974,8,Mid,Low,0.0,Petrol_Manual,Hyundai,60.0,0
1974,CAR_001975,Tata Harrier XZ BSIV,2019,1700000.0,10000.0,CNG,Individual,Manual,First Owner,1975,6,Premium,Low,0.0,Diesel_Manual,Tata,170.0,1
1975,CAR_001976,Nissan Terrano XL Plus 85 PS,2015,451000.0,100000.0,Diesel,Individual,Manual,First Owner,1976,10,Mid,High,0.0,Diesel_Manual,Nissan,4.51,1
1976,CAR_001977,Mahindra KUV 100 mFALCON G80 K8 5str,2016,370000.0,60000.0,Petrol,Individual,Manual,First Owner,1977,9,Mid,Medium,0.0,Petrol_Manual,Mahindra,6.17,0
1977,CAR_001978,Ford Endeavour 3.2 Titanium AT 4X4,2017,2700000.0,30000.0,Diesel,Individual,Automatic,First Owner,1978,8,Premium,Low,0.0,Diesel_Automatic,Ford,90.0,1
1978,CAR_001979,Renault KWID RXT Optional,2016,4461000.0,23000.0,Petrol,Individual,Manual,First Owner,1979,9,Low,Low,0.0,Petrol_Manual,Renault,12.17,0
1979,CAR_001980,Maruti Wagon R VXI BS IV,2017,350000.0,30000.0,Petrol,Individual,Manual,First Owner,1980,8,Mid,Low,0.0,Petrol_Manual,Maruti,11.67,0
1980,CAR_001981,Hyundai i10 Sportz,2011,250000.0,50000.0,Petrol,Individual,Manual,First Owner,1981,14,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
1981,CAR_001982,Maruti Alto 800 LXI,2014,204999.0,25000.0,Petrol,Individual,Manual,Second Owner,1982,11,Low,Low,0.0,Petrol_Manual,Maruti,8.2,0
1982,CAR_001983,Maruti Ignis 1.2 Delta BSIV,2019,500000.0,15000.0,Petrol,Individual,Manual,First Owner,1983,6,Mid,Low,0.0,Petrol_Manual,Maruti,33.33,0
1983,CAR_001984,Tata Indica Vista Aqua TDI BSIII,2009,150000.0,30000.0,Diesel,Individual,Manual,First Owner,1984,16,Low,Low,0.0,Diesel_Manual,Tata,5.0,1
1984,CAR_001985,Chevrolet Spark 1.0 LS,2010,175000.0,45000.0,Petrol,Individual,Manual,Second Owner,1985,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.89,0
1985,CAR_001986,Chevrolet Spark 1.0 LS,2009,170000.0,44800.0,Petrol,Individual,Manual,Second Owner,1986,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.79,0
1986,CAR_001987,Tata Manza Aura Safire,2010,200000.0,50000.0,Petrol,Individual,Manual,First Owner,1987,15,Low,Medium,0.0,Petrol_Manual,Tata,4.0,0
1987,CAR_001988,Hyundai Creta 1.4 E Plus,2018,800000.0,52000.0,Diesel,Individual,Manual,First Owner,1988,7,High,Medium,0.0,Diesel_Manual,Hyundai,15.38,1
1988,CAR_001989,Tata Hexa XTA,2017,1200000.0,50000.0,Diesel,Individual,Automatic,First Owner,1989,8,Premium,Medium,0.0,Diesel_Automatic,Tata,24.0,1
1989,CAR_001990,Toyota Corolla Altis 1.8 J,2008,400000.0,35000.0,LPG,Individual,Manual,Second Owner,1990,17,Mid,Medium,0.0,Petrol_Manual,Toyota,11.43,0
1990,CAR_001991,Maruti Ritz VDi,2010,148000.0,100000.0,Diesel,Individual,Manual,Second Owner,1991,15,Low,High,0.0,Diesel_Manual,Maruti,1.48,1
1991,CAR_001992,Tata Manza Club Class Quadrajet90 LS,2014,4461000.0,100000.0,Diesel,Individual,Manual,First Owner,1992,11,Low,High,0.0,Diesel_Manual,Tata,2.0,1
1992,CAR_001993,Maruti Swift Dzire VDI,2016,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,1993,9,Low,Medium,0.0,Diesel_Manual,Maruti,7.1,1
1993,CAR_001994,Maruti Wagon R VXI Minor,2007,80000.0,60000.0,Petrol,Individual,Manual,Third Owner,1994,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.33,0
1994,CAR_001995,Chevrolet Beat PS,2014,170000.0,77000.0,Petrol,Individual,Manual,First Owner,1995,11,Low,High,0.0,Petrol_Manual,Chevrolet,2.21,0
1995,CAR_001996,Hyundai Accent Executive,2011,150000.0,60000.0,Petrol,Individual,Manual,First Owner,1996,14,Low,Medium,0.0,Petrol_Manual,Hyundai,2.5,0
1996,CAR_001997,Tata Manza Aura (ABS) Safire BS IV,2010,110000.0,77073.0,Petrol,Individual,Manual,Second Owner,1997,15,Low,High,0.0,Petrol_Manual,Tata,1.43,0
1997,CAR_001998,Ford Figo Aspire 1.5 TDCi Titanium,2016,400000.0,70000.0,Diesel,Individual,Manual,First Owner,1998,9,Mid,Medium,0.0,Diesel_Manual,Ford,5.71,1
1998,CAR_001999,Hyundai i10 Magna LPG,2013,275000.0,90000.0,LPG,Individual,Manual,Second Owner,1999,12,Low,High,0.0,LPG_Manual,Hyundai,3.06,0
1999,CAR_002000,Toyota Innova 2.5 G (Diesel) 7 Seater,2013,800000.0,186000.0,Diesel,Individual,Manual,Third Owner,2000,12,High,Very High,0.0,Diesel_Manual,Toyota,4.3,1
2000,CAR_002001,Mahindra Scorpio M2DI,2014,550000.0,90000.0,Diesel,Individual,Manual,First Owner,2001,11,Mid,High,0.0,Diesel_Manual,Mahindra,6.11,1
2001,CAR_002002,Maruti Alto LX BSIII,2007,125000.0,100000.0,Petrol,Individual,Manual,Second Owner,2002,18,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
2002,CAR_002003,Chevrolet Beat LT,2010,4461000.0,80000.0,Petrol,Individual,Manual,Second Owner,2003,15,Low,High,0.0,Petrol_Manual,Chevrolet,1.69,0
2003,CAR_002004,Mahindra Scorpio LX BSIV,2012,349000.0,150000.0,Diesel,Individual,Manual,First Owner,2004,13,Mid,High,0.0,Diesel_Manual,Mahindra,2.33,1
2004,CAR_002005,Maruti Swift VDI BSIV,2014,458000.0,80000.0,Diesel,Individual,Manual,First Owner,2005,11,Mid,High,0.0,Diesel_Manual,Maruti,5.72,1
2005,CAR_002006,Maruti Alto K10 VXI,2018,300000.0,35000.0,Petrol,Individual,Manual,First Owner,2006,7,Low,Medium,0.0,Petrol_Manual,Maruti,8.57,0
2006,CAR_002007,Maruti Alto LXi,2008,95000.0,70000.0,Petrol,Individual,Manual,First Owner,2007,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.36,0
2007,CAR_002008,Maruti S-Cross Zeta DDiS 200 SH,2018,850000.0,50000.0,Diesel,Individual,Manual,First Owner,2008,7,Low,Medium,0.0,Diesel_Manual,Maruti,17.0,1
2008,CAR_002009,Ford Freestyle Titanium Diesel,2019,750000.0,25000.0,Electric,Individual,Manual,First Owner,2009,6,High,Low,0.0,Diesel_Manual,Ford,30.0,1
2009,CAR_002010,Ford Figo Diesel Celebration Edition,2013,190000.0,120000.0,Diesel,Individual,Manual,First Owner,2010,12,Low,High,0.0,Diesel_Manual,Ford,1.58,1
2010,CAR_002011,Maruti Wagon R LXI Minor,2009,150000.0,70000.0,Petrol,Individual,Manual,First Owner,2011,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.14,0
2011,CAR_002012,Hyundai Accent GLE,2009,114999.0,106000.0,Petrol,Individual,Manual,Third Owner,2012,16,Low,High,0.0,Petrol_Manual,Hyundai,1.08,0
2012,CAR_002013,Toyota Innova 2.0 VX 7 Seater,2010,320000.0,120000.0,Petrol,Individual,Manual,Third Owner,2013,15,Mid,High,0.0,Petrol_Manual,Toyota,2.67,0
2013,CAR_002014,Maruti Alto 800 CNG LXI Optional,2019,300000.0,120000.0,CNG,Individual,Manual,First Owner,2014,6,Low,High,0.0,CNG_Manual,Maruti,2.5,0
2014,CAR_002015,Maruti Ritz LXi,2016,275000.0,160000.0,Petrol,Individual,Manual,First Owner,2015,9,Low,Very High,0.0,Petrol_Manual,Maruti,1.72,0
2015,CAR_002016,Chevrolet Aveo 1.6 LT with ABS,2010,140000.0,120000.0,Petrol,Individual,Manual,Second Owner,2016,15,Low,High,0.0,Petrol_Manual,Chevrolet,1.17,0
2016,CAR_002017,Tata Altroz XZ,2020,830000.0,10000.0,Petrol,Individual,Manual,First Owner,2017,5,High,Low,0.0,Petrol_Manual,Tata,83.0,0
2017,CAR_002018,Volkswagen Polo SR Petrol 1.2L,2013,380000.0,50000.0,Petrol,Individual,Manual,First Owner,2018,12,Mid,Medium,0.0,Petrol_Manual,Volkswagen,7.6,0
2018,CAR_002019,Hyundai i10 Magna 1.2,2010,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,2019,15,Low,High,0.0,Petrol_Manual,Hyundai,1.67,0
2019,CAR_002020,Chevrolet Spark 1.0 LS,2008,50000.0,70000.0,Petrol,Individual,Manual,First Owner,2020,17,Low,Medium,0.0,Petrol_Manual,Chevrolet,0.71,0
2020,CAR_002021,Mahindra Bolero SLX,2008,155000.0,100000.0,Diesel,Individual,Manual,First Owner,2021,17,Low,High,0.0,Diesel_Manual,Mahindra,1.55,1
2021,CAR_002022,Mahindra Bolero SLX,2008,155000.0,60000.0,Petrol,Individual,Manual,First Owner,2022,17,Low,Medium,0.0,Diesel_Manual,Mahindra,1.55,1
2022,CAR_002023,Skoda Superb 1.8 TFSI MT,2010,300000.0,60000.0,Petrol,Individual,Manual,First Owner,2023,15,Low,High,0.0,Petrol_Manual,Skoda,3.75,0
2023,CAR_002024,Maruti Alto 800 LXI,2018,300000.0,16584.0,Diesel,Dealer,Manual,First Owner,2024,7,Low,Medium,0.0,Petrol_Manual,Maruti,18.09,0
2024,CAR_002025,Tata Tiago 2019-2020 XE Diesel,2016,385000.0,38000.0,Diesel,Dealer,Manual,First Owner,2025,9,Mid,Medium,0.0,Diesel_Manual,Tata,10.13,1
2025,CAR_002026,Maruti Wagon R LXI,2010,221000.0,51000.0,Petrol,Dealer,Manual,First Owner,2026,15,Low,Medium,0.0,Petrol_Manual,Maruti,4.33,0
2026,CAR_002027,Tata Tiago 1.05 Revotorq XE,2016,381000.0,38000.0,Diesel,Dealer,Manual,First Owner,2027,9,Mid,Medium,0.0,Diesel_Manual,Tata,10.03,1
2027,CAR_002028,Maruti Alto 800 LXI,2015,250999.0,35000.0,Petrol,Dealer,Manual,First Owner,2028,10,Low,Medium,0.0,Petrol_Manual,Maruti,7.17,0
2028,CAR_002029,Hyundai Santro LS zipPlus,2002,4461000.0,120000.0,Petrol,Individual,Manual,Third Owner,2029,23,Low,Medium,0.0,Petrol_Manual,Hyundai,0.5,0
2029,CAR_002030,Tata Tiago 1.05 Revotorq XE,2016,380000.0,38000.0,Diesel,Dealer,Manual,First Owner,2030,9,Mid,Medium,0.0,Diesel_Manual,Tata,10.0,1
2030,CAR_002031,Skoda Laura L n K 1.9 PD,2008,180000.0,144000.0,Diesel,Individual,Manual,Third Owner,2031,17,Low,High,0.0,Diesel_Manual,Skoda,1.25,1
2031,CAR_002032,Maruti Ciaz ZDi Plus,2017,751000.0,64000.0,Diesel,Dealer,Manual,First Owner,2032,8,High,Medium,0.0,Diesel_Manual,Maruti,11.73,1
2032,CAR_002033,Mahindra XUV500 AT W10 FWD,2015,1250000.0,60000.0,Diesel,Dealer,Automatic,First Owner,2033,10,Premium,Medium,0.0,Diesel_Automatic,Mahindra,20.83,1
2033,CAR_002034,Hyundai EON 1.0 Era Plus,2014,229999.0,65000.0,Petrol,Dealer,Manual,First Owner,2034,11,Low,Medium,0.0,Petrol_Manual,Hyundai,3.54,0
2034,CAR_002035,Tata Safari Storme EX,2019,1250000.0,24000.0,Diesel,Individual,Manual,First Owner,2035,6,Premium,Low,0.0,Diesel_Manual,Tata,52.08,1
2035,CAR_002036,Mahindra Scorpio VLX 2WD BSIV,2014,782000.0,58000.0,Diesel,Dealer,Manual,Second Owner,2036,11,High,Medium,0.0,Diesel_Manual,Mahindra,13.48,1
2036,CAR_002037,Mahindra Scorpio VLX 2.2 mHawk Airbag BSIV,2014,780000.0,58000.0,CNG,Dealer,Manual,Second Owner,2037,11,High,Medium,0.0,Diesel_Manual,Mahindra,13.45,1
2037,CAR_002038,Maruti Alto 800 LXI,2016,245000.0,68000.0,Petrol,Dealer,Manual,First Owner,2038,9,Low,Medium,0.0,Petrol_Manual,Maruti,3.6,0
2038,CAR_002039,Hyundai i20 1.4 CRDi Sportz,2011,321000.0,81257.0,LPG,Dealer,Manual,First Owner,2039,14,Mid,High,0.0,Diesel_Manual,Hyundai,3.95,1
2039,CAR_002040,Hyundai Grand i10 1.2 CRDi Sportz Option,2017,509999.0,44000.0,Diesel,Dealer,Manual,First Owner,2040,8,Mid,Medium,0.0,Diesel_Manual,Hyundai,11.59,1
2040,CAR_002041,Tata Nano CX SE,2015,110000.0,45000.0,Petrol,Dealer,Manual,First Owner,2041,10,Low,Medium,0.0,Petrol_Manual,Tata,2.44,0
2041,CAR_002042,Volkswagen Polo Diesel Highline 1.2L,2013,400000.0,75000.0,Diesel,Dealer,Manual,First Owner,2042,12,Low,Medium,0.0,Diesel_Manual,Volkswagen,5.33,1
2042,CAR_002043,Maruti Swift VXI,2013,4461000.0,90000.0,Petrol,Individual,Manual,First Owner,2043,12,Mid,High,0.0,Petrol_Manual,Maruti,4.11,0
2043,CAR_002044,Hyundai Verna CRDi ABS,2007,175000.0,80000.0,Diesel,Individual,Manual,Second Owner,2044,18,Low,High,0.0,Diesel_Manual,Hyundai,2.19,1
2044,CAR_002045,Mercedes-Benz E-Class 230,1998,1000000.0,35000.0,Electric,Individual,Automatic,Second Owner,2045,27,High,Medium,0.0,Petrol_Automatic,Mercedes-Benz,28.57,0
2045,CAR_002046,Maruti Alto 800 LXI,2013,92800.0,25000.0,Petrol,Individual,Manual,Second Owner,2046,12,Low,Low,0.0,Petrol_Manual,Maruti,3.71,0
2046,CAR_002047,Tata Xenon XT EX 4X2,2014,291000.0,90000.0,Diesel,Individual,Manual,First Owner,2047,11,Low,High,0.0,Diesel_Manual,Tata,3.23,1
2047,CAR_002048,Maruti Alto 800 LXI,2013,92800.0,25000.0,Petrol,Individual,Manual,Second Owner,2048,12,Low,Low,0.0,Petrol_Manual,Maruti,3.71,0
2048,CAR_002049,Hyundai Verna 1.6 SX CRDi (O),2013,335000.0,110000.0,Diesel,Individual,Manual,Third Owner,2049,12,Mid,High,0.0,Diesel_Manual,Hyundai,3.05,1
2049,CAR_002050,Maruti Omni 8 Seater BSIV,2013,170000.0,35000.0,Petrol,Individual,Manual,First Owner,2050,12,Low,Medium,0.0,Petrol_Manual,Maruti,4.86,0
2050,CAR_002051,Maruti Alto 800 LXI,2019,295000.0,10000.0,Petrol,Individual,Manual,First Owner,2051,6,Low,Low,0.0,Petrol_Manual,Maruti,29.5,0
2051,CAR_002052,Renault KWID RXL,2019,375000.0,10000.0,Diesel,Individual,Manual,First Owner,2052,6,Mid,Low,0.0,Petrol_Manual,Renault,37.5,0
2052,CAR_002053,Hyundai EON Era Plus,2012,215000.0,70000.0,Petrol,Individual,Manual,Second Owner,2053,13,Low,Medium,0.0,Petrol_Manual,Hyundai,3.07,0
2053,CAR_002054,Ford Fiesta 1.4 ZXi TDCi ABS,2009,4461000.0,110000.0,Diesel,Individual,Manual,Second Owner,2054,16,Low,High,0.0,Diesel_Manual,Ford,1.36,1
2054,CAR_002055,Maruti Alto LXI,2003,60000.0,5000.0,Petrol,Individual,Manual,Second Owner,2055,22,Low,Medium,0.0,Petrol_Manual,Maruti,12.0,0
2055,CAR_002056,Maruti Alto LX,2006,52000.0,120000.0,Petrol,Individual,Manual,Third Owner,2056,19,Low,High,0.0,Petrol_Manual,Maruti,0.43,0
2056,CAR_002057,Maruti Omni LPG CARGO BSIII W IMMOBILISER,2009,80000.0,90000.0,LPG,Individual,Manual,Second Owner,2057,16,Low,High,0.0,LPG_Manual,Maruti,0.89,0
2057,CAR_002058,Maruti Alto LX,2012,140000.0,70000.0,Petrol,Individual,Manual,Second Owner,2058,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
2058,CAR_002059,Hyundai Verna i (Petrol),2008,120000.0,90000.0,Petrol,Individual,Manual,Second Owner,2059,17,Low,High,0.0,Petrol_Manual,Hyundai,1.33,0
2059,CAR_002060,Ford Fiesta Classic 1.4 SXI Duratorq,2006,110000.0,120000.0,Diesel,Individual,Manual,Third Owner,2060,19,Low,High,0.0,Diesel_Manual,Ford,0.92,1
2060,CAR_002061,Tata New Safari Dicor EX 4X2 BS IV,2012,280000.0,110000.0,Diesel,Individual,Manual,Fourth & Above Owner,2061,13,Low,High,0.0,Diesel_Manual,Tata,2.55,1
2061,CAR_002062,Toyota Etios VD,2016,450000.0,50000.0,Diesel,Individual,Manual,Second Owner,2062,9,Mid,Medium,0.0,Diesel_Manual,Toyota,9.0,1
2062,CAR_002063,Mahindra XUV500 W7,2018,1150000.0,58000.0,CNG,Individual,Manual,First Owner,2063,7,Low,Medium,0.0,Diesel_Manual,Mahindra,19.83,1
2063,CAR_002064,Maruti Baleno Delta 1.3,2018,4461000.0,15000.0,Diesel,Individual,Manual,First Owner,2064,7,High,Low,0.0,Diesel_Manual,Maruti,43.33,1
2064,CAR_002065,Tata Zest Revotron 1.2T XE,2015,300000.0,60000.0,Petrol,Individual,Manual,Second Owner,2065,10,Low,Medium,0.0,Petrol_Manual,Tata,5.0,0
2065,CAR_002066,Tata Nano Twist XT,2015,73000.0,57000.0,Petrol,Individual,Manual,First Owner,2066,10,Low,Medium,0.0,Petrol_Manual,Tata,1.28,0
2066,CAR_002067,Maruti Swift VDI BSIV,2015,400000.0,87000.0,Diesel,Individual,Manual,Third Owner,2067,10,Mid,High,0.0,Diesel_Manual,Maruti,4.6,1
2067,CAR_002068,Renault KWID 1.0 RXT Optional,2017,300000.0,60000.0,Petrol,Individual,Manual,First Owner,2068,8,Low,Medium,0.0,Petrol_Manual,Renault,4.29,0
2068,CAR_002069,Maruti Wagon R AX,2012,300000.0,30000.0,LPG,Individual,Automatic,Second Owner,2069,13,Low,Low,0.0,Petrol_Automatic,Maruti,10.0,0
2069,CAR_002070,Maruti Swift Dzire VDI,2012,450000.0,80000.0,Diesel,Individual,Manual,First Owner,2070,13,Mid,High,0.0,Diesel_Manual,Maruti,5.62,1
2070,CAR_002071,Hyundai Santro Xing GLS,2009,200000.0,60000.0,Petrol,Individual,Manual,Second Owner,2071,16,Low,High,0.0,Petrol_Manual,Hyundai,2.0,0
2071,CAR_002072,Maruti Swift VXI,2019,580000.0,40000.0,Petrol,Individual,Manual,First Owner,2072,6,Mid,Medium,0.0,Petrol_Manual,Maruti,14.5,0
2072,CAR_002073,Hyundai Grand i10 1.2 Kappa Asta,2017,550000.0,3917.0,Petrol,Individual,Manual,First Owner,2073,8,Mid,Low,0.0,Petrol_Manual,Hyundai,140.41,0
2073,CAR_002074,Mahindra Scorpio SLX 2.6 Turbo 8 Str,2007,4461000.0,160000.0,Diesel,Individual,Manual,Second Owner,2074,18,Low,Medium,0.0,Diesel_Manual,Mahindra,1.19,1
2074,CAR_002075,Tata Tiago 1.05 Revotorq XT Option,2017,450000.0,35000.0,Diesel,Individual,Manual,First Owner,2075,8,Mid,Medium,0.0,Diesel_Manual,Tata,12.86,1
2075,CAR_002076,Chevrolet Spark 1.0 LS,2014,220000.0,70000.0,Petrol,Individual,Manual,First Owner,2076,11,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.14,0
2076,CAR_002077,Maruti SX4 ZXI MT BSIV,2012,310000.0,69069.0,Petrol,Dealer,Manual,First Owner,2077,13,Mid,Medium,0.0,Petrol_Manual,Maruti,4.49,0
2077,CAR_002078,Hyundai Elite i20 Sportz Plus BSIV,2019,650000.0,13500.0,Petrol,Individual,Manual,First Owner,2078,6,Low,Medium,0.0,Petrol_Manual,Hyundai,48.15,0
2078,CAR_002079,Maruti Wagon R Stingray VXI,2014,300000.0,60000.0,Petrol,Dealer,Manual,First Owner,2079,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.08,0
2079,CAR_002080,Hyundai Accent Executive,2013,200000.0,70000.0,Petrol,Individual,Manual,First Owner,2080,12,Low,Medium,0.0,Petrol_Manual,Hyundai,2.86,0
2080,CAR_002081,Honda Brio S MT,2012,285000.0,39039.0,Petrol,Dealer,Manual,First Owner,2081,13,Low,Medium,0.0,Petrol_Manual,Honda,7.3,0
2081,CAR_002082,Honda Brio Exclusive Edition,2014,335000.0,33033.0,Petrol,Dealer,Manual,First Owner,2082,11,Mid,Medium,0.0,Petrol_Manual,Honda,10.14,0
2082,CAR_002083,Hyundai Verna Transform SX VTVT,2011,370000.0,55168.0,Petrol,Individual,Manual,Second Owner,2083,14,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.71,0
2083,CAR_002084,Hyundai i20 1.2 Sportz,2013,375000.0,41041.0,Petrol,Dealer,Manual,First Owner,2084,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.14,0
2084,CAR_002085,Hyundai i10 Magna,2009,235000.0,67067.0,Petrol,Dealer,Manual,First Owner,2085,16,Low,Medium,0.0,Petrol_Manual,Hyundai,3.5,0
2085,CAR_002086,Hyundai i10 Sportz 1.2 AT,2013,4461000.0,66066.0,Petrol,Dealer,Automatic,First Owner,2086,12,Mid,Medium,0.0,Petrol_Automatic,Hyundai,5.22,0
2086,CAR_002087,Mahindra Quanto C8,2013,365000.0,82082.0,Diesel,Dealer,Manual,First Owner,2087,12,Mid,High,0.0,Diesel_Manual,Mahindra,4.45,1
2087,CAR_002088,Hyundai Grand i10 Sportz,2014,400000.0,60000.0,Petrol,Individual,Manual,First Owner,2088,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.33,0
2088,CAR_002089,Hyundai EON D Lite Plus,2015,250000.0,30000.0,Petrol,Individual,Manual,First Owner,2089,10,Low,Low,0.0,Petrol_Manual,Hyundai,8.33,0
2089,CAR_002090,Hyundai Verna SX CRDi AT,2012,465000.0,60000.0,Diesel,Dealer,Automatic,First Owner,2090,13,Mid,Medium,0.0,Diesel_Automatic,Hyundai,6.64,1
2090,CAR_002091,Renault KWID RXT,2017,300000.0,10000.0,Petrol,Individual,Manual,First Owner,2091,8,Low,Low,0.0,Petrol_Manual,Renault,30.0,0
2091,CAR_002092,Maruti Swift Dzire ZXI 1.2 BS IV,2018,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,2092,7,High,Medium,0.0,Petrol_Manual,Maruti,26.2,0
2092,CAR_002093,Maruti Wagon R LXI CNG,2016,350000.0,30000.0,CNG,Individual,Manual,Second Owner,2093,9,Mid,Low,0.0,CNG_Manual,Maruti,11.67,0
2093,CAR_002094,Hyundai i10 Sportz 1.2 AT,2015,500000.0,50000.0,Petrol,Dealer,Automatic,First Owner,2094,10,Mid,Medium,0.0,Petrol_Automatic,Hyundai,10.0,0
2094,CAR_002095,Hyundai Grand i10 AT Asta,2015,465000.0,63063.0,Petrol,Dealer,Automatic,First Owner,2095,10,Mid,Medium,0.0,Petrol_Automatic,Hyundai,7.37,0
2095,CAR_002096,Maruti Swift VDi BSIII W/ ABS,2008,4461000.0,70000.0,Diesel,Individual,Manual,First Owner,2096,17,Low,Medium,0.0,Diesel_Manual,Maruti,3.76,1
2096,CAR_002097,Maruti Omni MPI STD BSIV,2012,250000.0,90000.0,Petrol,Individual,Manual,First Owner,2097,13,Low,High,0.0,Petrol_Manual,Maruti,2.78,0
2097,CAR_002098,Hyundai Accent GLE 1,1999,60000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,2098,26,Low,High,0.0,Petrol_Manual,Hyundai,0.67,0
2098,CAR_002099,Hyundai Getz GLX,2005,70000.0,90000.0,Petrol,Individual,Manual,Third Owner,2099,20,Low,High,0.0,Petrol_Manual,Hyundai,0.78,0
2099,CAR_002100,Chevrolet Sail Hatchback 1.2 LS,2016,250000.0,30000.0,Petrol,Individual,Manual,First Owner,2100,9,Low,Low,0.0,Petrol_Manual,Chevrolet,8.33,0
2100,CAR_002101,Hyundai EON Magna Plus,2016,229999.0,80000.0,Petrol,Individual,Manual,First Owner,2101,9,Low,High,0.0,Petrol_Manual,Hyundai,2.87,0
2101,CAR_002102,Chevrolet Optra Magnum 2.0 LS,2011,195000.0,120000.0,Diesel,Individual,Manual,First Owner,2102,14,Low,High,0.0,Diesel_Manual,Chevrolet,1.62,1
2102,CAR_002103,Datsun RediGO S,2016,270000.0,60000.0,Electric,Individual,Manual,First Owner,2103,9,Low,Low,0.0,Petrol_Manual,Datsun,12.27,0
2103,CAR_002104,Honda WR-V i-DTEC VX,2017,650000.0,140000.0,Diesel,Individual,Manual,First Owner,2104,8,High,Medium,0.0,Diesel_Manual,Honda,4.64,1
2104,CAR_002105,Hyundai i20 Asta 1.4 CRDi,2014,370000.0,80000.0,Diesel,Individual,Manual,Second Owner,2105,11,Mid,High,0.0,Diesel_Manual,Hyundai,4.62,1
2105,CAR_002106,Maruti Swift VDI Optional,2016,509999.0,40000.0,Diesel,Individual,Manual,First Owner,2106,9,Mid,Medium,0.0,Diesel_Manual,Maruti,12.75,1
2106,CAR_002107,Maruti Zen Estilo VXI BSIV,2011,280000.0,33000.0,Petrol,Individual,Manual,First Owner,2107,14,Low,Medium,0.0,Petrol_Manual,Maruti,8.48,0
2107,CAR_002108,Maruti A-Star Vxi,2010,180000.0,40000.0,Petrol,Individual,Manual,First Owner,2108,15,Low,Medium,0.0,Petrol_Manual,Maruti,4.5,0
2108,CAR_002109,Renault KWID 1.0 RXT Optional,2018,300000.0,10500.0,Petrol,Individual,Manual,First Owner,2109,7,Low,Low,0.0,Petrol_Manual,Renault,28.57,0
2109,CAR_002110,Mahindra Bolero 2011-2019 SLX,2013,400000.0,107000.0,Diesel,Individual,Manual,First Owner,2110,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.74,1
2110,CAR_002111,Hyundai i10 Sportz 1.2,2010,210000.0,60000.0,Petrol,Individual,Manual,Second Owner,2111,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.5,0
2111,CAR_002112,Hyundai Santro Xing XG eRLX Euro III,2006,90000.0,120000.0,Petrol,Individual,Manual,Second Owner,2112,19,Low,High,0.0,Petrol_Manual,Hyundai,0.75,0
2112,CAR_002113,Maruti Wagon R LXI BS IV,2010,222000.0,70000.0,Petrol,Individual,Manual,First Owner,2113,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.17,0
2113,CAR_002114,Maruti Alto 800 LXI,2015,240000.0,50000.0,Diesel,Individual,Manual,Second Owner,2114,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.8,0
2114,CAR_002115,Toyota Etios Liva GD,2013,450000.0,70000.0,Diesel,Individual,Manual,Second Owner,2115,12,Mid,Medium,0.0,Diesel_Manual,Toyota,6.43,1
2115,CAR_002116,Ford Ecosport 1.5 Petrol Ambiente,2017,650000.0,6000.0,Petrol,Individual,Manual,First Owner,2116,8,High,Low,0.0,Petrol_Manual,Ford,108.33,0
2116,CAR_002117,Hyundai Verna SX Diesel,2010,200000.0,78000.0,Diesel,Dealer,Manual,Second Owner,2117,15,Low,High,0.0,Diesel_Manual,Hyundai,2.56,1
2117,CAR_002118,Hyundai Verna SX Diesel,2011,210000.0,72000.0,Diesel,Dealer,Manual,Third Owner,2118,14,Low,High,0.0,Diesel_Manual,Hyundai,2.92,1
2118,CAR_002119,Tata Nano Cx BSIV,2011,75000.0,9528.0,Petrol,Dealer,Manual,First Owner,2119,14,Low,Low,0.0,Petrol_Manual,Tata,7.87,0
2119,CAR_002120,Toyota Innova 2.5 G (Diesel) 8 Seater,2011,450000.0,135200.0,CNG,Dealer,Manual,First Owner,2120,14,Mid,High,0.0,Diesel_Manual,Toyota,3.33,1
2120,CAR_002121,Hyundai i20 1.2 Sportz,2011,220000.0,58000.0,Petrol,Dealer,Manual,Second Owner,2121,14,Low,Medium,0.0,Petrol_Manual,Hyundai,3.79,0
2121,CAR_002122,Hyundai i20 Magna 1.4 CRDi (Diesel),2013,325000.0,78000.0,Diesel,Dealer,Manual,Second Owner,2122,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.17,1
2122,CAR_002123,Tata Zest Revotron 1.2T XE,2018,450000.0,45000.0,Petrol,Individual,Manual,First Owner,2123,7,Low,Medium,0.0,Petrol_Manual,Tata,10.0,0
2123,CAR_002124,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2017,825000.0,35000.0,Diesel,Individual,Manual,First Owner,2124,8,High,Medium,0.0,Diesel_Manual,Ford,23.57,1
2124,CAR_002125,Hyundai Accent GLS,2007,125000.0,60000.0,Petrol,Dealer,Manual,Third Owner,2125,18,Low,High,0.0,Petrol_Manual,Hyundai,1.64,0
2125,CAR_002126,Mahindra Xylo D2 BS IV,2015,434999.0,100000.0,Diesel,Individual,Manual,Third Owner,2126,10,Mid,High,0.0,Diesel_Manual,Mahindra,4.35,1
2126,CAR_002127,Mahindra Scorpio SLE BSIV,2011,484999.0,99000.0,Diesel,Individual,Manual,Second Owner,2127,14,Mid,High,0.0,Diesel_Manual,Mahindra,4.9,1
2127,CAR_002128,Maruti Ritz VDi,2009,4461000.0,60000.0,Diesel,Individual,Manual,Third Owner,2128,16,Low,High,0.0,Diesel_Manual,Maruti,2.19,1
2128,CAR_002129,Honda Civic 1.8 S MT,2006,220000.0,120000.0,LPG,Individual,Manual,Third Owner,2129,19,Low,High,0.0,Petrol_Manual,Honda,1.83,0
2129,CAR_002130,Honda BR-V i-VTEC VX MT,2020,1250000.0,60000.0,Petrol,Dealer,Manual,First Owner,2130,5,Low,Low,0.0,Petrol_Manual,Honda,1136.36,0
2130,CAR_002131,Maruti Omni E MPI STD BS IV,2017,215000.0,34000.0,Petrol,Dealer,Manual,First Owner,2131,8,Low,Medium,0.0,Petrol_Manual,Maruti,6.32,0
2131,CAR_002132,Mahindra XUV500 W6 2WD,2014,625000.0,60000.0,Diesel,Dealer,Manual,First Owner,2132,11,High,Medium,0.0,Diesel_Manual,Mahindra,9.19,1
2132,CAR_002133,Ford EcoSport 1.5 Diesel Titanium BSIV,2018,615000.0,60000.0,Diesel,Individual,Manual,First Owner,2133,7,High,Low,0.0,Diesel_Manual,Ford,41.0,1
2133,CAR_002134,Chevrolet Beat Diesel PS,2012,130000.0,58000.0,Diesel,Dealer,Manual,First Owner,2134,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.24,1
2134,CAR_002135,Tata Indigo CS eLX BS IV,2013,175000.0,50300.0,Diesel,Individual,Manual,First Owner,2135,12,Low,Medium,0.0,Diesel_Manual,Tata,3.48,1
2135,CAR_002136,Tata Indigo CR4,2011,150000.0,70000.0,Diesel,Individual,Manual,Second Owner,2136,14,Low,Medium,0.0,Diesel_Manual,Tata,2.14,1
2136,CAR_002137,Tata Indica Vista TDI LS,2011,195000.0,60000.0,Diesel,Individual,Manual,Third Owner,2137,14,Low,Medium,0.0,Diesel_Manual,Tata,3.25,1
2137,CAR_002138,Maruti Ertiga 1.5 VDI,2020,550000.0,90000.0,Diesel,Individual,Manual,First Owner,2138,5,Mid,High,0.0,Diesel_Manual,Maruti,6.11,1
2138,CAR_002139,Isuzu D-Max V-Cross Standard,2018,1500000.0,40000.0,Diesel,Individual,Manual,First Owner,2139,7,Premium,Medium,0.0,Diesel_Manual,Isuzu,37.5,1
2139,CAR_002140,Skoda Superb 1.8 TFSI MT,2010,400000.0,80000.0,Petrol,Individual,Manual,Second Owner,2140,15,Mid,High,0.0,Petrol_Manual,Skoda,5.0,0
2140,CAR_002141,Nissan Kicks XV Premium D BSIV,2019,1350000.0,15000.0,Diesel,Individual,Manual,First Owner,2141,6,Premium,Low,0.0,Diesel_Manual,Nissan,90.0,1
2141,CAR_002142,Maruti Swift 1.2 DLX,2015,217000.0,45000.0,Petrol,Individual,Manual,First Owner,2142,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.82,0
2142,CAR_002143,Maruti Swift VDI BSIV,2011,290000.0,130000.0,Diesel,Individual,Manual,Second Owner,2143,14,Low,High,0.0,Diesel_Manual,Maruti,2.23,1
2143,CAR_002144,Mahindra Bolero Power Plus SLX,2018,700000.0,80000.0,Diesel,Individual,Manual,First Owner,2144,7,High,High,0.0,Diesel_Manual,Mahindra,8.75,1
2144,CAR_002145,Maruti Alto 800 Base,2014,175000.0,40000.0,Petrol,Individual,Manual,First Owner,2145,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.38,0
2145,CAR_002146,Volkswagen Ameo 1.5 TDI Highline 16 Alloy,2017,575000.0,80000.0,Diesel,Individual,Manual,First Owner,2146,8,Mid,High,0.0,Diesel_Manual,Volkswagen,7.19,1
2146,CAR_002147,Maruti Ritz LDi,2016,380000.0,100000.0,Diesel,Individual,Manual,Second Owner,2147,9,Mid,High,0.0,Diesel_Manual,Maruti,3.8,1
2147,CAR_002148,Mahindra Xylo D4 BSIV,2013,350000.0,151624.0,Diesel,Individual,Manual,Second Owner,2148,12,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.31,1
2148,CAR_002149,Renault Duster 85PS Diesel RxL Optional,2012,450000.0,160000.0,Diesel,Individual,Manual,First Owner,2149,13,Mid,Medium,0.0,Diesel_Manual,Renault,2.81,1
2149,CAR_002150,Hyundai Santro AT CNG,2005,130000.0,110000.0,CNG,Individual,Manual,Second Owner,2150,20,Low,Medium,0.0,CNG_Manual,Hyundai,1.18,0
2150,CAR_002151,Maruti Swift VDI BSIV,2015,320000.0,74820.0,Diesel,Individual,Manual,First Owner,2151,10,Mid,High,0.0,Diesel_Manual,Maruti,4.28,1
2151,CAR_002152,Mahindra XUV500 W6 2WD,2014,550000.0,82000.0,Diesel,Individual,Manual,First Owner,2152,11,Mid,High,0.0,Diesel_Manual,Mahindra,6.71,1
2152,CAR_002153,Maruti Wagon R VXI BS IV,2015,370000.0,50000.0,Petrol,Individual,Manual,Second Owner,2153,10,Low,Medium,0.0,Petrol_Manual,Maruti,7.4,0
2153,CAR_002154,Chevrolet Beat LT,2010,114999.0,70000.0,Petrol,Individual,Manual,Second Owner,2154,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.64,0
2154,CAR_002155,Ford Ecosport Sports Petrol,2020,4461000.0,1000.0,Electric,Individual,Manual,First Owner,2155,5,Premium,Low,0.0,Petrol_Manual,Ford,1100.0,0
2155,CAR_002156,Hyundai Tucson CRDi,2006,170000.0,170000.0,Diesel,Individual,Manual,First Owner,2156,19,Low,Very High,0.0,Diesel_Manual,Hyundai,1.0,1
2156,CAR_002157,Tata Indigo TDI,2008,80000.0,150000.0,Diesel,Individual,Manual,Third Owner,2157,17,Low,High,0.0,Diesel_Manual,Tata,0.53,1
2157,CAR_002158,Ford EcoSport 1.5 Diesel Titanium BSIV,2019,950000.0,17000.0,Diesel,Individual,Manual,First Owner,2158,6,High,Low,0.0,Diesel_Manual,Ford,55.88,1
2158,CAR_002159,Maruti Swift Dzire VDI,2014,425000.0,150000.0,Diesel,Individual,Manual,First Owner,2159,11,Mid,High,0.0,Diesel_Manual,Maruti,2.83,1
2159,CAR_002160,Hyundai i20 Asta Option 1.4 CRDi,2015,500000.0,80000.0,Diesel,Individual,Manual,First Owner,2160,10,Mid,High,0.0,Diesel_Manual,Hyundai,6.25,1
2160,CAR_002161,Hyundai Santro Xing XG eRLX Euro III,2005,70000.0,170000.0,Petrol,Individual,Manual,Third Owner,2161,20,Low,Very High,0.0,Petrol_Manual,Hyundai,0.41,0
2161,CAR_002162,Honda Amaze S i-Dtech,2013,349000.0,120000.0,Diesel,Individual,Manual,First Owner,2162,12,Low,High,0.0,Diesel_Manual,Honda,2.91,1
2162,CAR_002163,Hyundai i20 1.2 Sportz,2012,330000.0,60000.0,Petrol,Individual,Manual,Third Owner,2163,13,Mid,High,0.0,Petrol_Manual,Hyundai,3.67,0
2163,CAR_002164,Hyundai Santro Xing GLS,2010,120000.0,90000.0,Petrol,Individual,Manual,Third Owner,2164,15,Low,Medium,0.0,Petrol_Manual,Hyundai,1.33,0
2164,CAR_002165,Hyundai Verna 1.6 SX CRDi (O),2013,430000.0,100000.0,Diesel,Individual,Manual,Second Owner,2165,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.3,1
2165,CAR_002166,Maruti Swift VDI BSIV,2014,390000.0,90000.0,Diesel,Individual,Manual,First Owner,2166,11,Low,High,0.0,Diesel_Manual,Maruti,4.33,1
2166,CAR_002167,Nissan Micra Diesel XV Primo,2012,225000.0,129000.0,Diesel,Individual,Manual,First Owner,2167,13,Low,High,0.0,Diesel_Manual,Nissan,1.74,1
2167,CAR_002168,Mahindra Scorpio VLS AT 2.2 mHAWK,2010,350000.0,186000.0,Diesel,Individual,Automatic,Second Owner,2168,15,Mid,Very High,0.0,Diesel_Automatic,Mahindra,1.88,1
2168,CAR_002169,Tata Sumo LX,2009,100000.0,66778.0,Diesel,Individual,Manual,First Owner,2169,16,Low,High,0.0,Diesel_Manual,Tata,1.5,1
2169,CAR_002170,Hyundai EON Era Plus,2014,190000.0,50000.0,Petrol,Individual,Manual,First Owner,2170,11,Low,Medium,0.0,Petrol_Manual,Hyundai,3.8,0
2170,CAR_002171,Hyundai i20 Sportz 1.4 CRDi,2014,4461000.0,110000.0,Diesel,Individual,Manual,Second Owner,2171,11,Mid,High,0.0,Diesel_Manual,Hyundai,2.86,1
2171,CAR_002172,Hyundai Grand i10 CRDi Magna,2015,300000.0,60000.0,Diesel,Individual,Manual,Second Owner,2172,10,Low,Medium,0.0,Diesel_Manual,Hyundai,5.0,1
2172,CAR_002173,Renault KWID RXT,2016,280000.0,40000.0,Petrol,Individual,Manual,First Owner,2173,9,Low,Medium,0.0,Petrol_Manual,Renault,7.0,0
2173,CAR_002174,Hyundai Tucson 2.0 e-VGT 2WD MT,2017,1650000.0,55000.0,Diesel,Dealer,Manual,First Owner,2174,8,Premium,Medium,0.0,Diesel_Manual,Hyundai,30.0,1
2174,CAR_002175,Hyundai EON Era Plus,2015,295000.0,60000.0,Petrol,Dealer,Manual,Second Owner,2175,10,Low,High,0.0,Petrol_Manual,Hyundai,14.05,0
2175,CAR_002176,Jaguar XF 5.0 Litre V8 Petrol,2012,2050000.0,66363.0,Petrol,Dealer,Automatic,Second Owner,2176,13,Premium,Medium,0.0,Petrol_Automatic,Jaguar,30.89,0
2176,CAR_002177,Hyundai Creta 1.6 VTVT AT SX Plus,2018,4461000.0,11700.0,Petrol,Dealer,Automatic,First Owner,2177,7,Premium,Low,0.0,Petrol_Automatic,Hyundai,126.07,0
2177,CAR_002178,Hyundai Verna VTVT 1.6 AT SX Option,2017,1100000.0,10000.0,Petrol,Individual,Automatic,First Owner,2178,8,Premium,Low,0.0,Petrol_Automatic,Hyundai,110.0,0
2178,CAR_002179,Mercedes-Benz GL-Class 350 CDI Blue Efficiency,2014,4400000.0,100000.0,Diesel,Individual,Automatic,Second Owner,2179,11,Premium,High,0.0,Diesel_Automatic,Mercedes-Benz,44.0,1
2179,CAR_002180,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,2180,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
2180,CAR_002181,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,2181,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
2181,CAR_002182,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,2182,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
2182,CAR_002183,Volkswagen Polo GTI,2017,850000.0,20000.0,Petrol,Dealer,Automatic,Second Owner,2183,8,High,Low,0.0,Petrol_Automatic,Volkswagen,42.5,0
2183,CAR_002184,Renault Pulse RxL,2015,4461000.0,61585.0,Diesel,Trustmark Dealer,Manual,First Owner,2184,10,Mid,Medium,0.0,Diesel_Manual,Renault,6.33,1
2184,CAR_002185,Maruti Celerio VXI AMT BSIV,2016,450000.0,39415.0,Diesel,Trustmark Dealer,Automatic,Second Owner,2185,9,Low,Medium,0.0,Petrol_Automatic,Maruti,11.42,0
2185,CAR_002186,Honda Brio V MT,2014,425000.0,29654.0,Petrol,Trustmark Dealer,Manual,First Owner,2186,11,Mid,Low,0.0,Petrol_Manual,Honda,14.33,0
2186,CAR_002187,Maruti Baleno Alpha 1.3,2016,4461000.0,64672.0,Diesel,Trustmark Dealer,Manual,First Owner,2187,9,Low,Medium,0.0,Diesel_Manual,Maruti,11.91,1
2187,CAR_002188,Hyundai Creta 1.6 SX Automatic Diesel,2015,1150000.0,54634.0,Diesel,Trustmark Dealer,Automatic,Second Owner,2188,10,Low,Medium,0.0,Diesel_Automatic,Hyundai,21.05,1
2188,CAR_002189,Honda City i VTEC V,2015,775000.0,60000.0,Petrol,Trustmark Dealer,Manual,First Owner,2189,10,Low,Medium,0.0,Petrol_Manual,Honda,11.65,0
2189,CAR_002190,Hyundai Elite i20 Asta Option CVT BSIV,2019,720000.0,3000.0,Petrol,Individual,Automatic,First Owner,2190,6,High,Low,0.0,Petrol_Automatic,Hyundai,240.0,0
2190,CAR_002191,Toyota Innova Crysta 2.4 GX AT,2018,1725000.0,23974.0,Diesel,Dealer,Automatic,Second Owner,2191,7,Premium,Low,0.0,Diesel_Automatic,Toyota,71.95,1
2191,CAR_002192,Tata Tiago 1.2 Revotron XZ,2018,4461000.0,10000.0,Petrol,Individual,Manual,First Owner,2192,7,Mid,Low,0.0,Petrol_Manual,Tata,53.9,0
2192,CAR_002193,Maruti Celerio ZXI,2017,420000.0,50000.0,Petrol,Individual,Manual,First Owner,2193,8,Mid,Medium,0.0,Petrol_Manual,Maruti,8.4,0
2193,CAR_002194,Maruti Omni E 8 Str STD,2001,85000.0,70000.0,CNG,Individual,Manual,Third Owner,2194,24,Low,Medium,0.0,Petrol_Manual,Maruti,1.21,0
2194,CAR_002195,OpelCorsa 1.6Gls,2004,142000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,2195,21,Low,High,0.0,Petrol_Manual,OpelCorsa,1.95,0
2195,CAR_002196,Hyundai EON D Lite,2014,180000.0,76000.0,Petrol,Individual,Manual,Second Owner,2196,11,Low,High,0.0,Petrol_Manual,Hyundai,2.37,0
2196,CAR_002197,Chevrolet Tavera Neo 3 Max 9 Str BSIII,2014,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,2197,11,Mid,High,0.0,Diesel_Manual,Chevrolet,3.33,1
2197,CAR_002198,Mahindra Scorpio VLX 2WD ABS AT BSIII,2009,360000.0,170000.0,Diesel,Individual,Automatic,Third Owner,2198,16,Mid,Very High,0.0,Diesel_Automatic,Mahindra,2.12,1
2198,CAR_002199,Hyundai Santro Xing XG,2003,85000.0,110000.0,Petrol,Individual,Manual,Second Owner,2199,22,Low,High,0.0,Petrol_Manual,Hyundai,0.77,0
2199,CAR_002200,Honda Brio 1.2 S MT,2016,350000.0,63400.0,Petrol,Individual,Manual,First Owner,2200,9,Mid,Medium,0.0,Petrol_Manual,Honda,5.52,0
2200,CAR_002201,Datsun GO Plus T BSIV,2015,380000.0,25000.0,LPG,Individual,Manual,First Owner,2201,10,Mid,Low,0.0,Petrol_Manual,Datsun,15.2,0
2201,CAR_002202,Hyundai i10 Era,2009,140000.0,80000.0,Petrol,Individual,Manual,Third Owner,2202,16,Low,High,0.0,Petrol_Manual,Hyundai,1.75,0
2202,CAR_002203,Maruti Wagon R VXI BS IV,2012,200000.0,90000.0,Petrol,Individual,Manual,First Owner,2203,13,Low,High,0.0,Petrol_Manual,Maruti,2.22,0
2203,CAR_002204,Hyundai Santro Xing GL,2011,150000.0,157000.0,Petrol,Individual,Manual,First Owner,2204,14,Low,Very High,0.0,Petrol_Manual,Hyundai,0.96,0
2204,CAR_002205,Tata Indica Vista Quadrajet LS,2012,150000.0,80000.0,Diesel,Individual,Manual,Second Owner,2205,13,Low,High,0.0,Diesel_Manual,Tata,1.88,1
2205,CAR_002206,Maruti 800 Std BSII,2006,90000.0,40000.0,Petrol,Individual,Manual,Second Owner,2206,19,Low,Medium,0.0,Petrol_Manual,Maruti,2.25,0
2206,CAR_002207,Mahindra Bolero SLX 2WD,2007,300000.0,100000.0,Diesel,Individual,Manual,Second Owner,2207,18,Low,High,0.0,Diesel_Manual,Mahindra,3.0,1
2207,CAR_002208,Renault KWID RXT,2017,280000.0,60000.0,Petrol,Individual,Manual,First Owner,2208,8,Low,Medium,0.0,Petrol_Manual,Renault,5.6,0
2208,CAR_002209,Maruti Alto K10 2010-2014 VXI,2011,4461000.0,40000.0,Petrol,Individual,Manual,First Owner,2209,14,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
2209,CAR_002210,Chevrolet Aveo U-VA 1.2 LT WO ABS Airbag,2009,110000.0,38500.0,Petrol,Individual,Manual,Second Owner,2210,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.86,0
2210,CAR_002211,Hyundai i10 Magna,2008,140000.0,103921.0,Petrol,Individual,Manual,Second Owner,2211,17,Low,High,0.0,Petrol_Manual,Hyundai,1.35,0
2211,CAR_002212,Hyundai Creta 1.4 EX Diesel,2020,1050000.0,10000.0,Diesel,Individual,Manual,First Owner,2212,5,Low,Low,0.0,Diesel_Manual,Hyundai,105.0,1
2212,CAR_002213,Honda City i-VTEC VX,2018,1150000.0,14825.0,Petrol,Dealer,Manual,First Owner,2213,7,Premium,Low,0.0,Petrol_Manual,Honda,77.57,0
2213,CAR_002214,Honda BR-V i-DTEC VX MT,2016,910000.0,43377.0,Electric,Dealer,Manual,First Owner,2214,9,High,Medium,0.0,Diesel_Manual,Honda,20.98,1
2214,CAR_002215,Ford Figo Diesel Titanium,2010,160000.0,60000.0,Diesel,Dealer,Manual,First Owner,2215,15,Low,High,0.0,Diesel_Manual,Ford,1.56,1
2215,CAR_002216,Maruti Zen LX,2004,4461000.0,79000.0,Petrol,Individual,Manual,Second Owner,2216,21,Low,High,0.0,Petrol_Manual,Maruti,1.33,0
2216,CAR_002217,Skoda Octavia Elegance 2.0 TDI AT,2014,1200000.0,135000.0,Diesel,Individual,Manual,Third Owner,2217,11,Premium,High,0.0,Diesel_Automatic,Skoda,8.89,1
2217,CAR_002218,Renault KWID 1.0 RXT Optional,2018,300000.0,10500.0,Petrol,Individual,Manual,First Owner,2218,7,Low,Low,0.0,Petrol_Manual,Renault,28.57,0
2218,CAR_002219,Maruti Wagon R LXI,2004,70000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,2219,21,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
2219,CAR_002220,Chevrolet Beat Diesel,2012,160000.0,110000.0,Diesel,Individual,Manual,First Owner,2220,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.45,1
2220,CAR_002221,Chevrolet Beat Diesel LS,2012,160000.0,50000.0,Diesel,Individual,Manual,First Owner,2221,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,3.2,1
2221,CAR_002222,Chevrolet Sail Hatchback LT ABS,2013,225000.0,80000.0,Diesel,Individual,Manual,First Owner,2222,12,Low,High,0.0,Diesel_Manual,Chevrolet,2.81,1
2222,CAR_002223,Honda Jazz 1.5 VX i DTEC,2018,790000.0,19571.0,Diesel,Dealer,Manual,First Owner,2223,7,High,Low,0.0,Diesel_Manual,Honda,40.37,1
2223,CAR_002224,Honda City 1.3 EXI,2002,4461000.0,100000.0,Petrol,Individual,Manual,First Owner,2224,23,Low,High,0.0,Petrol_Manual,Honda,1.45,0
2224,CAR_002225,Honda City i-VTEC ZX,2018,1200000.0,29600.0,Petrol,Dealer,Manual,First Owner,2225,7,Premium,High,0.0,Petrol_Manual,Honda,40.54,0
2225,CAR_002226,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2018,930000.0,60000.0,Diesel,Individual,Manual,First Owner,2226,7,High,High,0.0,Diesel_Manual,Ford,46.5,1
2226,CAR_002227,Hyundai EON D Lite,2016,210000.0,30000.0,Petrol,Individual,Manual,First Owner,2227,9,Low,High,0.0,Petrol_Manual,Hyundai,7.0,0
2227,CAR_002228,Hyundai Getz 1.5 CRDi GVS,2008,4461000.0,154000.0,Diesel,Individual,Manual,Third Owner,2228,17,Low,Very High,0.0,Diesel_Manual,Hyundai,0.71,1
2228,CAR_002229,Maruti Swift LXI,2011,175000.0,70000.0,Petrol,Individual,Manual,First Owner,2229,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
2229,CAR_002230,Maruti Vitara Brezza ZDi Plus AMT Dual Tone,2018,950000.0,15000.0,Diesel,Individual,Manual,First Owner,2230,7,High,Low,0.0,Diesel_Automatic,Maruti,63.33,1
2230,CAR_002231,Renault Lodgy Stepway 85PS RXZ 8S,2017,650000.0,40000.0,Diesel,Individual,Manual,First Owner,2231,8,High,Medium,0.0,Diesel_Manual,Renault,16.25,1
2231,CAR_002232,Maruti Swift VXI,2019,550000.0,51000.0,Petrol,Individual,Manual,First Owner,2232,6,Mid,Medium,0.0,Petrol_Manual,Maruti,10.78,0
2232,CAR_002233,Maruti Alto LX BSIII,2008,85000.0,120000.0,Petrol,Individual,Manual,Second Owner,2233,17,Low,High,0.0,Petrol_Manual,Maruti,0.71,0
2233,CAR_002234,Maruti Celerio LXI MT BSIV,2019,340000.0,60000.0,Petrol,Individual,Manual,First Owner,2234,6,Mid,Medium,0.0,Petrol_Manual,Maruti,6.8,0
2234,CAR_002235,Renault Captur 1.5 Diesel RXT,2017,825000.0,13500.0,Diesel,Individual,Manual,First Owner,2235,8,High,Low,0.0,Diesel_Manual,Renault,61.11,1
2235,CAR_002236,Toyota Etios 1.5 V,2016,509999.0,35000.0,Petrol,Individual,Manual,First Owner,2236,9,Mid,Medium,0.0,Petrol_Manual,Toyota,14.57,0
2236,CAR_002237,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Petrol,Dealer,Manual,Second Owner,2237,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
2237,CAR_002238,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,2238,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
2238,CAR_002239,Audi A4 3.0 TDI Quattro,2013,4461000.0,86000.0,Diesel,Dealer,Automatic,First Owner,2239,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
2239,CAR_002240,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,2240,6,Low,Low,0.0,Diesel_Automatic,BMW,165.0,1
2240,CAR_002241,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,2241,9,Low,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
2241,CAR_002242,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,2242,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
2242,CAR_002243,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Manual,First Owner,2243,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
2243,CAR_002244,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,2244,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
2244,CAR_002245,Maruti Ertiga VXI,2015,4461000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,2245,10,Low,Low,0.0,Petrol_Manual,Maruti,52.44,0
2245,CAR_002246,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,2246,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
2246,CAR_002247,Chevrolet Beat LT Option,2016,239000.0,60000.0,Petrol,Dealer,Manual,First Owner,2247,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
2247,CAR_002248,Toyota Fortuner 4x2 AT,2017,2600000.0,60000.0,Diesel,Trustmark Dealer,Automatic,First Owner,2248,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
2248,CAR_002249,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,2249,10,Low,Medium,0.0,Diesel_Manual,Maruti,16.31,1
2249,CAR_002250,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,2250,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
2250,CAR_002251,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,2251,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
2251,CAR_002252,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,2252,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
2252,CAR_002253,Skoda Laura Ambiente 2.0 TDI CR MT,2012,4461000.0,52000.0,Diesel,Dealer,Manual,First Owner,2253,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
2253,CAR_002254,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,2254,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
2254,CAR_002255,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,2255,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
2255,CAR_002256,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,2256,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
2256,CAR_002257,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,2257,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
2257,CAR_002258,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,2258,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
2258,CAR_002259,BMW X5 xDrive 30d xLine,2019,4950000.0,30000.0,Diesel,Dealer,Automatic,First Owner,2259,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
2259,CAR_002260,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,CNG,Individual,Manual,First Owner,2260,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
2260,CAR_002261,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,2261,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
2261,CAR_002262,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Manual,First Owner,2262,8,Mid,High,0.0,Petrol_Automatic,Hyundai,27.65,0
2262,CAR_002263,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,2263,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
2263,CAR_002264,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,2264,10,Low,Low,0.0,Petrol_Manual,Maruti,52.44,0
2264,CAR_002265,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,2265,8,Mid,High,0.0,Petrol_Automatic,Hyundai,49.48,0
2265,CAR_002266,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,2266,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
2266,CAR_002267,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Automatic,First Owner,2267,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
2267,CAR_002268,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,60000.0,Diesel,Trustmark Dealer,Manual,First Owner,2268,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
2268,CAR_002269,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,2269,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
2269,CAR_002270,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,2270,12,Low,Medium,0.0,Diesel_Automatic,Audi,22.22,1
2270,CAR_002271,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,2271,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
2271,CAR_002272,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,2272,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
2272,CAR_002273,Hyundai Verna 1.6 VTVT SX,2015,760000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,2273,10,High,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
2273,CAR_002274,Maruti Swift Dzire VDI,2017,600000.0,46507.0,Diesel,Trustmark Dealer,Manual,First Owner,2274,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
2274,CAR_002275,Maruti Baleno Alpha 1.3,2017,650000.0,70000.0,Diesel,Individual,Manual,First Owner,2275,8,High,Medium,0.0,Diesel_Manual,Maruti,9.29,1
2275,CAR_002276,Chevrolet Spark 1.0 LS,2011,140000.0,60000.0,Petrol,Individual,Manual,Second Owner,2276,14,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.33,0
2276,CAR_002277,Maruti Alto LXi,2011,150000.0,70000.0,Petrol,Individual,Manual,First Owner,2277,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.14,0
2277,CAR_002278,Datsun GO Plus A Option Petrol,2018,450000.0,10000.0,Petrol,Individual,Manual,Second Owner,2278,7,Mid,Low,0.0,Petrol_Manual,Datsun,45.0,0
2278,CAR_002279,Hyundai Accent CRDi,2006,170000.0,245244.0,Diesel,Individual,Manual,Fourth & Above Owner,2279,19,Low,Very High,0.0,Diesel_Manual,Hyundai,0.69,1
2279,CAR_002280,Renault KWID 1.0 RXT Optional,2017,400000.0,60000.0,Petrol,Individual,Manual,First Owner,2280,8,Mid,Medium,0.0,Petrol_Manual,Renault,6.67,0
2280,CAR_002281,Tata Tiago 1.2 Revotron XE,2016,300000.0,30000.0,Petrol,Individual,Manual,First Owner,2281,9,Low,Low,0.0,Petrol_Manual,Tata,10.0,0
2281,CAR_002282,Maruti Alto 800 Base,2012,250000.0,8000.0,Petrol,Individual,Manual,First Owner,2282,13,Low,Low,0.0,Petrol_Manual,Maruti,31.25,0
2282,CAR_002283,Tata Tigor 1.2 Revotron XM,2019,500000.0,60000.0,Petrol,Individual,Manual,First Owner,2283,6,Mid,Low,0.0,Petrol_Manual,Tata,50.0,0
2283,CAR_002284,Hyundai Accent CRDi,2002,85000.0,100000.0,LPG,Individual,Manual,Third Owner,2284,23,Low,High,0.0,Diesel_Manual,Hyundai,0.85,1
2284,CAR_002285,Nissan Sunny Diesel XL,2012,300000.0,110000.0,Electric,Individual,Manual,First Owner,2285,13,Low,High,0.0,Diesel_Manual,Nissan,2.73,1
2285,CAR_002286,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,2286,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
2286,CAR_002287,Mahindra XUV500 AT W8 FWD,2015,740000.0,45000.0,Diesel,Dealer,Automatic,First Owner,2287,10,High,Medium,0.0,Diesel_Automatic,Mahindra,16.44,1
2287,CAR_002288,Hyundai Santro Xing XG,2005,70000.0,68500.0,Petrol,Dealer,Manual,First Owner,2288,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.02,0
2288,CAR_002289,Hyundai Santro Sportz AMT,2019,484999.0,60000.0,Petrol,Dealer,Automatic,First Owner,2289,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,96.86,0
2289,CAR_002290,Nissan Micra Active XV S,2013,164000.0,30000.0,Petrol,Individual,Manual,First Owner,2290,12,Low,Low,0.0,Petrol_Manual,Nissan,5.47,0
2290,CAR_002291,Honda City 1.5 V AT,2008,4461000.0,70000.0,Petrol,Individual,Automatic,First Owner,2291,17,Low,High,0.0,Petrol_Automatic,Honda,2.0,0
2291,CAR_002292,Mercedes-Benz E-Class E250 CDI Elegance,2011,999000.0,49600.0,Diesel,Dealer,Automatic,First Owner,2292,14,High,Medium,0.0,Diesel_Automatic,Mercedes-Benz,20.14,1
2292,CAR_002293,Maruti Alto LXI,2005,56000.0,23000.0,Petrol,Individual,Manual,Second Owner,2293,20,Low,Low,0.0,Petrol_Manual,Maruti,2.43,0
2293,CAR_002294,BMW 7 Series 730Ld,2006,4461000.0,30000.0,Diesel,Dealer,Automatic,First Owner,2294,19,Premium,Low,0.0,Diesel_Automatic,BMW,35.0,1
2294,CAR_002295,Hyundai Verna 1.6 VTVT,2010,190000.0,38000.0,Petrol,Dealer,Manual,First Owner,2295,15,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
2295,CAR_002296,Audi Q5 3.0 TDI Quattro Technology,2018,3899000.0,60000.0,Diesel,Dealer,Automatic,First Owner,2296,7,Premium,Low,0.0,Diesel_Automatic,Audi,177.23,1
2296,CAR_002297,Hyundai i10 Sportz 1.2,2011,235000.0,60000.0,Petrol,Dealer,Manual,First Owner,2297,14,Low,Medium,0.0,Petrol_Manual,Hyundai,5.45,0
2297,CAR_002298,Datsun GO Plus T,2017,350000.0,10171.0,Petrol,Dealer,Manual,First Owner,2298,8,Mid,Low,0.0,Petrol_Manual,Datsun,34.41,0
2298,CAR_002299,Renault Duster 110PS Diesel RxL,2015,465000.0,41123.0,Diesel,Dealer,Manual,First Owner,2299,10,Mid,Medium,0.0,Diesel_Manual,Renault,11.31,1
2299,CAR_002300,Toyota Camry Hybrid 2.5,2017,1900000.0,20118.0,Petrol,Dealer,Automatic,First Owner,2300,8,Premium,Low,0.0,Petrol_Automatic,Toyota,94.44,0
2300,CAR_002301,Honda City i DTec SV,2014,450000.0,40000.0,Diesel,Dealer,Manual,First Owner,2301,11,Mid,Medium,0.0,Diesel_Manual,Honda,11.25,1
2301,CAR_002302,Volkswagen Jetta 2.0L TDI Highline,2015,790000.0,52517.0,Diesel,Dealer,Manual,First Owner,2302,10,High,Medium,0.0,Diesel_Manual,Volkswagen,15.04,1
2302,CAR_002303,Mahindra Marazzo M2 8Str,2019,900000.0,20000.0,Diesel,Individual,Manual,First Owner,2303,6,High,Low,0.0,Diesel_Manual,Mahindra,45.0,1
2303,CAR_002304,Ford Ecosport 1.5 DV5 MT Titanium Optional,2014,450000.0,99117.0,Diesel,Dealer,Manual,First Owner,2304,11,Mid,High,0.0,Diesel_Manual,Ford,4.54,1
2304,CAR_002305,Chevrolet Spark 1.0 LS,2009,135000.0,90000.0,Petrol,Individual,Manual,First Owner,2305,16,Low,High,0.0,Petrol_Manual,Chevrolet,1.5,0
2305,CAR_002306,Maruti Celerio ZXI MT BSIV,2019,425000.0,3700.0,Petrol,Individual,Manual,First Owner,2306,6,Mid,High,0.0,Petrol_Manual,Maruti,114.86,0
2306,CAR_002307,Tata Indica Vista Terra 1.4 TDI,2010,100000.0,80000.0,Diesel,Individual,Manual,First Owner,2307,15,Low,High,0.0,Diesel_Manual,Tata,1.25,1
2307,CAR_002308,Ford EcoSport 1.5 TDCi Titanium BSIV,2015,650000.0,100000.0,Diesel,Individual,Manual,Second Owner,2308,10,Low,High,0.0,Diesel_Manual,Ford,6.5,1
2308,CAR_002309,Tata Indica Vista Aqua 1.4 TDI,2011,90000.0,54000.0,Diesel,Individual,Manual,First Owner,2309,14,Low,Medium,0.0,Diesel_Manual,Tata,1.67,1
2309,CAR_002310,Hyundai EON Era Plus,2017,250000.0,19000.0,Petrol,Individual,Manual,First Owner,2310,8,Low,Low,0.0,Petrol_Manual,Hyundai,13.16,0
2310,CAR_002311,Maruti Swift Dzire VDI,2014,300000.0,120000.0,Diesel,Individual,Manual,First Owner,2311,11,Low,High,0.0,Diesel_Manual,Maruti,2.5,1
2311,CAR_002312,Maruti Wagon R VXI BS IV,2014,275000.0,30000.0,Petrol,Individual,Manual,First Owner,2312,11,Low,High,0.0,Petrol_Manual,Maruti,9.17,0
2312,CAR_002313,Honda BRV i-VTEC V MT,2017,600000.0,43500.0,Petrol,Individual,Manual,First Owner,2313,8,Mid,Medium,0.0,Petrol_Manual,Honda,13.79,0
2313,CAR_002314,Renault KWID RXT,2016,270000.0,60000.0,Petrol,Individual,Manual,Second Owner,2314,9,Low,High,0.0,Petrol_Manual,Renault,4.5,0
2314,CAR_002315,Renault KWID RXT Optional,2016,240000.0,70000.0,Petrol,Individual,Manual,Second Owner,2315,9,Low,Medium,0.0,Petrol_Manual,Renault,3.43,0
2315,CAR_002316,Skoda Octavia Elegance 2.0 TDI AT,2014,1200000.0,135000.0,Diesel,Individual,Automatic,Third Owner,2316,11,Low,High,0.0,Diesel_Automatic,Skoda,8.89,1
2316,CAR_002317,Chevrolet Beat Diesel LS,2012,160000.0,50000.0,Diesel,Individual,Manual,First Owner,2317,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,3.2,1
2317,CAR_002318,Honda Jazz 1.5 VX i DTEC,2018,790000.0,19571.0,Diesel,Dealer,Manual,First Owner,2318,7,High,Low,0.0,Diesel_Manual,Honda,40.37,1
2318,CAR_002319,Honda City i-VTEC ZX,2018,1200000.0,29600.0,Petrol,Dealer,Manual,First Owner,2319,7,Premium,Low,0.0,Petrol_Manual,Honda,40.54,0
2319,CAR_002320,Maruti Swift LXI,2011,175000.0,70000.0,CNG,Individual,Manual,First Owner,2320,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
2320,CAR_002321,Maruti Vitara Brezza ZDi Plus AMT Dual Tone,2018,950000.0,15000.0,Diesel,Individual,Automatic,First Owner,2321,7,High,Low,0.0,Diesel_Automatic,Maruti,63.33,1
2321,CAR_002322,Maruti Celerio LXI MT BSIV,2019,340000.0,50000.0,Petrol,Individual,Manual,First Owner,2322,6,Mid,Medium,0.0,Petrol_Manual,Maruti,6.8,0
2322,CAR_002323,Renault Captur 1.5 Diesel RXT,2017,825000.0,13500.0,Diesel,Individual,Manual,First Owner,2323,8,Low,Low,0.0,Diesel_Manual,Renault,61.11,1
2323,CAR_002324,Audi A4 30 TFSI Technology,2018,3100000.0,22000.0,LPG,Individual,Automatic,First Owner,2324,7,Premium,Low,0.0,Petrol_Automatic,Audi,140.91,0
2324,CAR_002325,Honda Amaze VX Diesel BSIV,2018,780000.0,25000.0,Diesel,Dealer,Manual,First Owner,2325,7,High,Low,0.0,Diesel_Manual,Honda,31.2,1
2325,CAR_002326,Toyota Corolla Altis 1.8 VL AT,2010,350000.0,80000.0,Petrol,Individual,Automatic,Third Owner,2326,15,Mid,High,0.0,Petrol_Automatic,Toyota,4.38,0
2326,CAR_002327,Honda Amaze VX Petrol BSIV,2018,690000.0,39000.0,Petrol,Dealer,Manual,First Owner,2327,7,Low,High,0.0,Petrol_Manual,Honda,17.69,0
2327,CAR_002328,Maruti Alto 800 VXI,2016,245000.0,60000.0,Petrol,Individual,Manual,First Owner,2328,9,Low,Medium,0.0,Petrol_Manual,Maruti,4.08,0
2328,CAR_002329,Honda Amaze VX Diesel BSIV,2018,790000.0,49000.0,Diesel,Dealer,Manual,First Owner,2329,7,High,Medium,0.0,Diesel_Manual,Honda,16.12,1
2329,CAR_002330,Honda Amaze VX i-VTEC,2018,680000.0,48600.0,Petrol,Dealer,Manual,First Owner,2330,7,High,Medium,0.0,Petrol_Manual,Honda,13.99,0
2330,CAR_002331,Hyundai Grand i10 Asta Option,2017,540000.0,20000.0,Petrol,Individual,Manual,Second Owner,2331,8,Mid,Low,0.0,Petrol_Manual,Hyundai,27.0,0
2331,CAR_002332,Mahindra KUV 100 mFALCON G80 K2,2016,425000.0,60000.0,Petrol,Individual,Manual,First Owner,2332,9,Mid,Low,0.0,Petrol_Manual,Mahindra,17.0,0
2332,CAR_002333,Maruti Zen Estilo Sports,2008,140000.0,60000.0,Petrol,Individual,Manual,First Owner,2333,17,Low,High,0.0,Petrol_Manual,Maruti,1.17,0
2333,CAR_002334,Maruti Ertiga ZDI Plus,2019,1100000.0,60000.0,Diesel,Individual,Manual,First Owner,2334,6,Premium,Low,0.0,Diesel_Manual,Maruti,36.67,1
2334,CAR_002335,Renault Lodgy 85PS RxL,2015,715000.0,35000.0,Diesel,Individual,Manual,First Owner,2335,10,High,Medium,0.0,Diesel_Manual,Renault,20.43,1
2335,CAR_002336,Maruti Swift VXI,2013,370000.0,60000.0,Petrol,Individual,Manual,First Owner,2336,12,Mid,High,0.0,Petrol_Manual,Maruti,4.11,0
2336,CAR_002337,Hyundai Verna CRDi ABS,2007,175000.0,80000.0,Diesel,Individual,Manual,Second Owner,2337,18,Low,High,0.0,Diesel_Manual,Hyundai,2.19,1
2337,CAR_002338,Mercedes-Benz E-Class 230,1998,1000000.0,35000.0,Electric,Individual,Automatic,Second Owner,2338,27,High,Medium,0.0,Petrol_Automatic,Mercedes-Benz,28.57,0
2338,CAR_002339,Maruti Alto 800 LXI,2013,4461000.0,25000.0,Petrol,Individual,Manual,Second Owner,2339,12,Low,Low,0.0,Petrol_Manual,Maruti,3.71,0
2339,CAR_002340,Tata Xenon XT EX 4X2,2014,291000.0,90000.0,Diesel,Individual,Manual,First Owner,2340,11,Low,High,0.0,Diesel_Manual,Tata,3.23,1
2340,CAR_002341,Maruti Alto 800 LXI,2013,92800.0,60000.0,Petrol,Individual,Manual,Second Owner,2341,12,Low,Low,0.0,Petrol_Manual,Maruti,3.71,0
2341,CAR_002342,Renault KWID 1.0 RXT Optional,2016,350000.0,20000.0,Petrol,Individual,Manual,First Owner,2342,9,Mid,Low,0.0,Petrol_Manual,Renault,17.5,0
2342,CAR_002343,Maruti Omni 5 Str STD,1998,65000.0,60000.0,Petrol,Dealer,Manual,Third Owner,2343,27,Low,Medium,0.0,Petrol_Manual,Maruti,1.08,0
2343,CAR_002344,Maruti Swift Dzire VDI,2015,650000.0,58000.0,Diesel,Individual,Manual,First Owner,2344,10,High,Medium,0.0,Diesel_Manual,Maruti,11.21,1
2344,CAR_002345,Maruti Wagon R LX BS IV,2014,350000.0,60000.0,Diesel,Dealer,Manual,First Owner,2345,11,Low,High,0.0,Petrol_Manual,Maruti,8.75,0
2345,CAR_002346,Maruti Wagon R LXI,2005,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,2346,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.07,0
2346,CAR_002347,Hyundai i20 Asta (o) 1.4 CRDi (Diesel),2012,440000.0,70000.0,Diesel,Dealer,Manual,First Owner,2347,13,Low,Medium,0.0,Diesel_Manual,Hyundai,6.29,1
2347,CAR_002348,Hyundai Santro Xing GLS,2013,270000.0,60000.0,Petrol,Individual,Manual,First Owner,2348,12,Low,Medium,0.0,Petrol_Manual,Hyundai,4.5,0
2348,CAR_002349,Toyota Innova 2.5 V Diesel 7-seater,2014,1150000.0,135000.0,Diesel,Dealer,Manual,First Owner,2349,11,Premium,High,0.0,Diesel_Manual,Toyota,8.52,1
2349,CAR_002350,Maruti Omni 8 Seater BSII,2015,238000.0,70000.0,Petrol,Dealer,Manual,First Owner,2350,10,Low,High,0.0,Petrol_Manual,Maruti,3.4,0
2350,CAR_002351,Mercedes-Benz B Class B180 Sports,2013,1100000.0,40000.0,Petrol,Individual,Automatic,Second Owner,2351,12,Premium,High,0.0,Petrol_Automatic,Mercedes-Benz,27.5,0
2351,CAR_002352,Mahindra XUV500 W10 2WD,2015,1200000.0,75000.0,Diesel,Dealer,Manual,Second Owner,2352,10,Premium,High,0.0,Diesel_Manual,Mahindra,16.0,1
2352,CAR_002353,Skoda Fabia 1.2L Diesel Ambiente,2010,260000.0,150000.0,Diesel,Individual,Manual,First Owner,2353,15,Low,High,0.0,Diesel_Manual,Skoda,1.73,1
2353,CAR_002354,Toyota Etios VD,2012,350000.0,60000.0,Diesel,Individual,Manual,Third Owner,2354,13,Mid,High,0.0,Diesel_Manual,Toyota,2.5,1
2354,CAR_002355,Toyota Etios Liva VD,2015,480000.0,70000.0,CNG,Individual,Manual,First Owner,2355,10,Mid,Medium,0.0,Diesel_Manual,Toyota,6.86,1
2355,CAR_002356,Toyota Etios Liva GD,2012,345000.0,137250.0,Diesel,Individual,Manual,Second Owner,2356,13,Mid,High,0.0,Diesel_Manual,Toyota,2.51,1
2356,CAR_002357,Mahindra Scorpio M2DI,2013,500000.0,110000.0,Diesel,Individual,Manual,Second Owner,2357,12,Mid,High,0.0,Diesel_Manual,Mahindra,4.55,1
2357,CAR_002358,Hyundai Grand i10 Asta Option,2017,595000.0,27000.0,Petrol,Dealer,Manual,First Owner,2358,8,Mid,Low,0.0,Petrol_Manual,Hyundai,22.04,0
2358,CAR_002359,Hyundai Verna CRDi 1.6 SX Option,2019,1295000.0,60000.0,Diesel,Dealer,Manual,First Owner,2359,6,Premium,Low,0.0,Diesel_Manual,Hyundai,143.89,1
2359,CAR_002360,Volkswagen Vento 1.5 TDI Highline BSIV,2019,1350000.0,5400.0,Diesel,Dealer,Manual,Test Drive Car,2360,6,Premium,Low,0.0,Diesel_Manual,Volkswagen,250.0,1
2360,CAR_002361,Renault KWID Climber 1.0 MT Opt BSIV,2020,541000.0,1000.0,Petrol,Dealer,Manual,Test Drive Car,2361,5,Mid,Low,0.0,Petrol_Manual,Renault,541.0,0
2361,CAR_002362,Hyundai Santro Asta,2018,515000.0,16000.0,Petrol,Dealer,Manual,First Owner,2362,7,Mid,Low,0.0,Petrol_Manual,Hyundai,32.19,0
2362,CAR_002363,Ford Figo Aspire Titanium Plus Diesel,2019,894999.0,13000.0,Diesel,Dealer,Manual,Test Drive Car,2363,6,High,Low,0.0,Diesel_Manual,Ford,68.85,1
2363,CAR_002364,Renault Pulse RxZ Optional,2012,325000.0,47000.0,Diesel,Dealer,Manual,First Owner,2364,13,Low,High,0.0,Diesel_Manual,Renault,6.91,1
2364,CAR_002365,Maruti Ciaz 1.4 Alpha,2018,925000.0,22000.0,Petrol,Individual,Manual,First Owner,2365,7,High,Low,0.0,Petrol_Manual,Maruti,42.05,0
2365,CAR_002366,Nissan Terrano XL P,2017,844999.0,60000.0,Petrol,Individual,Manual,First Owner,2366,8,High,Low,0.0,Petrol_Manual,Nissan,28.17,0
2366,CAR_002367,Maruti Ciaz 1.4 Alpha,2018,4461000.0,20000.0,LPG,Individual,Manual,First Owner,2367,7,High,Low,0.0,Petrol_Manual,Maruti,46.25,0
2367,CAR_002368,Tata Manza Club Class Quadrajet90 LX,2013,300000.0,60000.0,Diesel,Individual,Manual,First Owner,2368,12,Low,Medium,0.0,Diesel_Manual,Tata,5.0,1
2368,CAR_002369,Toyota Camry M/t,2008,650000.0,75000.0,Petrol,Individual,Manual,Second Owner,2369,17,High,High,0.0,Petrol_Manual,Toyota,8.67,0
2369,CAR_002370,Maruti Alto K10 LXI,2010,175000.0,60000.0,Petrol,Dealer,Manual,First Owner,2370,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.73,0
2370,CAR_002371,Maruti SX4 S Cross DDiS 320 Delta,2015,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,2371,10,Mid,High,0.0,Diesel_Manual,Maruti,4.06,1
2371,CAR_002372,Renault Duster 85PS Diesel RxZ,2017,600000.0,60000.0,Diesel,Individual,Manual,First Owner,2372,8,Mid,High,0.0,Diesel_Manual,Renault,6.67,1
2372,CAR_002373,Maruti Swift ZXI Plus,2018,4461000.0,30000.0,Petrol,Individual,Manual,First Owner,2373,7,High,Low,0.0,Petrol_Manual,Maruti,23.33,0
2373,CAR_002374,Toyota Corolla H3,2003,95000.0,120000.0,Petrol,Individual,Automatic,Second Owner,2374,22,Low,High,0.0,Petrol_Automatic,Toyota,0.79,0
2374,CAR_002375,Maruti Celerio ZXI,2019,490000.0,20000.0,Petrol,Dealer,Manual,First Owner,2375,6,Mid,Low,0.0,Petrol_Manual,Maruti,24.5,0
2375,CAR_002376,Tata Safari DICOR 2.2 EX 4x2,2013,380000.0,100000.0,Diesel,Individual,Manual,First Owner,2376,12,Mid,High,0.0,Diesel_Manual,Tata,3.8,1
2376,CAR_002377,Hyundai Grand i10 1.2 Kappa Sportz Dual Tone,2018,550000.0,31000.0,Petrol,Dealer,Manual,First Owner,2377,7,Mid,Medium,0.0,Petrol_Manual,Hyundai,17.74,0
2377,CAR_002378,Maruti Alto 800 LXI,2013,225000.0,47000.0,Petrol,Dealer,Manual,First Owner,2378,12,Low,Medium,0.0,Petrol_Manual,Maruti,4.79,0
2378,CAR_002379,Tata Nexon 1.2 Revotron XM,2018,570000.0,11200.0,Petrol,Individual,Manual,First Owner,2379,7,Low,Low,0.0,Petrol_Manual,Tata,50.89,0
2379,CAR_002380,Maruti Alto 800 LXI,2018,305000.0,35000.0,Petrol,Dealer,Manual,First Owner,2380,7,Mid,High,0.0,Petrol_Manual,Maruti,8.71,0
2380,CAR_002381,Maruti Wagon R Duo Lxi,2010,4461000.0,50000.0,Electric,Dealer,Manual,First Owner,2381,15,Low,Medium,0.0,LPG_Manual,Maruti,4.8,0
2381,CAR_002382,Maruti Ertiga SHVS VDI,2015,700000.0,55000.0,Diesel,Dealer,Manual,First Owner,2382,10,High,Medium,0.0,Diesel_Manual,Maruti,12.73,1
2382,CAR_002383,Maruti Omni E MPI STD BS IV,2018,288000.0,20000.0,Petrol,Dealer,Manual,First Owner,2383,7,Low,Low,0.0,Petrol_Manual,Maruti,14.4,0
2383,CAR_002384,Mahindra Bolero B4,2017,800000.0,60000.0,Diesel,Dealer,Manual,First Owner,2384,8,High,High,0.0,Diesel_Manual,Mahindra,10.67,1
2384,CAR_002385,Maruti Celerio VXI,2017,430000.0,93000.0,Petrol,Dealer,Manual,First Owner,2385,8,Mid,High,0.0,Petrol_Manual,Maruti,4.62,0
2385,CAR_002386,Chevrolet Beat Diesel LT Option,2009,195000.0,57000.0,Diesel,Dealer,Manual,Second Owner,2386,16,Low,Medium,0.0,Diesel_Manual,Chevrolet,3.42,1
2386,CAR_002387,Maruti Eeco Smiles 5 Seater AC,2014,325000.0,30000.0,Petrol,Dealer,Manual,First Owner,2387,11,Mid,Low,0.0,Petrol_Manual,Maruti,10.83,0
2387,CAR_002388,Renault Duster 85PS Diesel RxE,2013,4461000.0,73000.0,Diesel,Dealer,Manual,First Owner,2388,12,Mid,High,0.0,Diesel_Manual,Renault,6.16,1
2388,CAR_002389,Maruti Swift Dzire VDI,2016,630000.0,60000.0,Diesel,Dealer,Manual,First Owner,2389,9,Low,Medium,0.0,Diesel_Manual,Maruti,9.69,1
2389,CAR_002390,Maruti Swift 1.3 VXi,2009,250000.0,60000.0,Petrol,Dealer,Manual,Second Owner,2390,16,Low,Medium,0.0,Petrol_Manual,Maruti,4.03,0
2390,CAR_002391,Hyundai Santro GLS I - Euro II,2007,135000.0,60000.0,Petrol,Dealer,Manual,First Owner,2391,18,Low,High,0.0,Petrol_Manual,Hyundai,1.59,0
2391,CAR_002392,Chevrolet Spark 1.0 PS,2010,110000.0,73000.0,Petrol,Individual,Manual,First Owner,2392,15,Low,High,0.0,Petrol_Manual,Chevrolet,1.51,0
2392,CAR_002393,Maruti Swift ZDi,2012,350000.0,55000.0,Diesel,Dealer,Manual,First Owner,2393,13,Mid,Medium,0.0,Diesel_Manual,Maruti,6.36,1
2393,CAR_002394,Maruti Omni E MPI STD BS IV,2016,240000.0,5800.0,Petrol,Individual,Manual,Second Owner,2394,9,Low,Low,0.0,Petrol_Manual,Maruti,41.38,0
2394,CAR_002395,Toyota Innova 2.5 V Diesel 8-seater,2009,350000.0,350000.0,Diesel,Individual,Manual,First Owner,2395,16,Mid,Very High,0.0,Diesel_Manual,Toyota,1.0,1
2395,CAR_002396,Maruti Swift ZXI Plus,2018,600000.0,30000.0,Petrol,Individual,Manual,First Owner,2396,7,Mid,Low,0.0,Petrol_Manual,Maruti,20.0,0
2396,CAR_002397,Maruti Swift Dzire VDI,2018,750000.0,20000.0,Diesel,Individual,Manual,First Owner,2397,7,High,Low,0.0,Diesel_Manual,Maruti,37.5,1
2397,CAR_002398,Hyundai EON Era Plus,2018,270000.0,20000.0,Petrol,Individual,Manual,First Owner,2398,7,Low,High,0.0,Petrol_Manual,Hyundai,13.5,0
2398,CAR_002399,Maruti Alto LXi,2007,90000.0,100000.0,Petrol,Individual,Manual,Second Owner,2399,18,Low,High,0.0,Petrol_Manual,Maruti,0.9,0
2399,CAR_002400,Maruti Wagon R Stingray VXI,2013,350000.0,65000.0,Petrol,Individual,Manual,First Owner,2400,12,Mid,Medium,0.0,Petrol_Manual,Maruti,5.38,0
2400,CAR_002401,Ford Figo Diesel Celebration Edition,2013,285000.0,90000.0,Diesel,Individual,Manual,Second Owner,2401,12,Low,High,0.0,Diesel_Manual,Ford,3.17,1
2401,CAR_002402,Toyota Innova 2.5 E Diesel MS 7-seater,2011,4461000.0,267000.0,Diesel,Individual,Manual,Second Owner,2402,14,High,Very High,0.0,Diesel_Manual,Toyota,2.49,1
2402,CAR_002403,Mahindra Scorpio 2.6 CRDe,2005,175000.0,250000.0,Diesel,Individual,Manual,Second Owner,2403,20,Low,Very High,0.0,Diesel_Manual,Mahindra,0.7,1
2403,CAR_002404,Mahindra Xylo D2,2011,350000.0,140000.0,Diesel,Individual,Manual,Second Owner,2404,14,Mid,High,0.0,Diesel_Manual,Mahindra,2.5,1
2404,CAR_002405,Honda City i-VTEC CVT ZX,2018,1225000.0,22000.0,Petrol,Dealer,Automatic,First Owner,2405,7,Premium,Low,0.0,Petrol_Automatic,Honda,55.68,0
2405,CAR_002406,Honda Jazz Select Edition Active,2010,285000.0,60000.0,Petrol,Dealer,Manual,Second Owner,2406,15,Low,Medium,0.0,Petrol_Manual,Honda,4.75,0
2406,CAR_002407,Honda City i-VTEC VX,2018,1000000.0,28635.0,Petrol,Dealer,Manual,First Owner,2407,7,High,Low,0.0,Petrol_Manual,Honda,34.92,0
2407,CAR_002408,Honda Amaze VX Diesel BSIV,2018,780000.0,32114.0,Diesel,Dealer,Manual,First Owner,2408,7,High,Medium,0.0,Diesel_Manual,Honda,24.29,1
2408,CAR_002409,Maruti Celerio VXI,2014,300000.0,82000.0,Petrol,Dealer,Manual,Second Owner,2409,11,Low,High,0.0,Petrol_Manual,Maruti,3.66,0
2409,CAR_002410,Maruti Swift VXI BSIII,2006,160000.0,95149.0,Petrol,Dealer,Manual,Second Owner,2410,19,Low,High,0.0,Petrol_Manual,Maruti,1.68,0
2410,CAR_002411,Honda Amaze VX i-VTEC,2018,625000.0,68458.0,Petrol,Dealer,Manual,First Owner,2411,7,High,Medium,0.0,Petrol_Manual,Honda,9.13,0
2411,CAR_002412,Maruti Ritz VXi,2010,4461000.0,42000.0,Petrol,Individual,Manual,First Owner,2412,15,Low,Medium,0.0,Petrol_Manual,Maruti,6.31,0
2412,CAR_002413,Honda City i DTEC V,2014,515000.0,90000.0,Diesel,Individual,Manual,Second Owner,2413,11,Mid,High,0.0,Diesel_Manual,Honda,5.72,1
2413,CAR_002414,Hyundai EON D Lite,2016,4461000.0,30000.0,Petrol,Individual,Manual,First Owner,2414,9,Low,Low,0.0,Petrol_Manual,Hyundai,7.0,0
2414,CAR_002415,Maruti Alto LXi,2009,140000.0,80000.0,Petrol,Individual,Manual,Second Owner,2415,16,Low,High,0.0,Petrol_Manual,Maruti,1.75,0
2415,CAR_002416,Toyota Innova 2.5 GX (Diesel) 7 Seater,2015,940000.0,128000.0,Diesel,Individual,Manual,First Owner,2416,10,High,High,0.0,Diesel_Manual,Toyota,7.34,1
2416,CAR_002417,Nissan Terrano XV Premium 110 PS,2013,750000.0,60000.0,CNG,Individual,Manual,First Owner,2417,12,High,High,0.0,Diesel_Manual,Nissan,9.38,1
2417,CAR_002418,Maruti Baleno Alpha 1.3,2016,610000.0,100000.0,Diesel,Individual,Manual,First Owner,2418,9,High,High,0.0,Diesel_Manual,Maruti,6.1,1
2418,CAR_002419,Nissan Sunny XV D Premium Leather,2015,450000.0,50000.0,Diesel,Individual,Manual,First Owner,2419,10,Mid,Medium,0.0,Diesel_Manual,Nissan,9.0,1
2419,CAR_002420,Toyota Etios Liva GD,2012,409999.0,56000.0,Diesel,Individual,Manual,First Owner,2420,13,Mid,Medium,0.0,Diesel_Manual,Toyota,7.32,1
2420,CAR_002421,Hyundai EON 1.0 Era Plus,2018,100000.0,60000.0,Petrol,Individual,Manual,First Owner,2421,7,Low,Low,0.0,Petrol_Manual,Hyundai,4.69,0
2421,CAR_002422,Hyundai i20 Active 1.4 SX,2016,650000.0,100000.0,Diesel,Individual,Manual,First Owner,2422,9,High,High,0.0,Diesel_Manual,Hyundai,6.5,1
2422,CAR_002423,Mahindra TUV 300 T8,2017,650000.0,80000.0,Diesel,Individual,Manual,First Owner,2423,8,High,High,0.0,Diesel_Manual,Mahindra,8.12,1
2423,CAR_002424,Hyundai Getz 1.3 GLS,2010,4461000.0,105546.0,Petrol,Individual,Manual,First Owner,2424,15,Low,High,0.0,Petrol_Manual,Hyundai,1.42,0
2424,CAR_002425,Toyota Etios Liva 1.2 V,2018,500000.0,10000.0,Petrol,Individual,Manual,First Owner,2425,7,Mid,Low,0.0,Petrol_Manual,Toyota,50.0,0
2425,CAR_002426,Toyota Innova 2.5 GX 7 STR BSIV,2012,1000000.0,80000.0,Diesel,Individual,Manual,Second Owner,2426,13,High,High,0.0,Diesel_Manual,Toyota,12.5,1
2426,CAR_002427,Hyundai i20 Magna Optional 1.2,2013,340000.0,35000.0,Petrol,Individual,Manual,First Owner,2427,12,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.71,0
2427,CAR_002428,Hyundai Elite i20 Asta Option BSIV,2019,800000.0,11240.0,Petrol,Individual,Manual,First Owner,2428,6,High,Low,0.0,Petrol_Manual,Hyundai,71.17,0
2428,CAR_002429,Toyota Innova 2.5 G (Diesel) 8 Seater,2013,1010000.0,120000.0,Diesel,Individual,Manual,First Owner,2429,12,Premium,High,0.0,Diesel_Manual,Toyota,8.42,1
2429,CAR_002430,Toyota Etios GD,2011,350000.0,120000.0,Diesel,Individual,Manual,Third Owner,2430,14,Mid,High,0.0,Diesel_Manual,Toyota,2.92,1
2430,CAR_002431,Maruti Swift VDI BSIV,2016,595000.0,70000.0,Diesel,Individual,Manual,First Owner,2431,9,Mid,Medium,0.0,Diesel_Manual,Maruti,8.5,1
2431,CAR_002432,Mahindra Scorpio VLX 2WD BSIV,2014,4461000.0,130000.0,Diesel,Individual,Manual,Third Owner,2432,11,High,High,0.0,Diesel_Manual,Mahindra,5.38,1
2432,CAR_002433,Renault KWID RXL,2019,265000.0,40000.0,Petrol,Individual,Manual,First Owner,2433,6,Low,Medium,0.0,Petrol_Manual,Renault,6.62,0
2433,CAR_002434,Maruti Alto K10 VXI,2015,320000.0,60000.0,Petrol,Individual,Manual,Second Owner,2434,10,Mid,Medium,0.0,Petrol_Manual,Maruti,5.33,0
2434,CAR_002435,Honda City 1.5 S MT,2011,310000.0,110000.0,Petrol,Individual,Manual,First Owner,2435,14,Mid,High,0.0,Petrol_Manual,Honda,2.82,0
2435,CAR_002436,Honda City i DTEC VX,2014,700000.0,130000.0,LPG,Individual,Manual,First Owner,2436,11,Low,High,0.0,Diesel_Manual,Honda,5.38,1
2436,CAR_002437,Maruti Alto LXi BSIII,2009,130000.0,80000.0,Petrol,Individual,Manual,Second Owner,2437,16,Low,High,0.0,Petrol_Manual,Maruti,1.62,0
2437,CAR_002438,Maruti Swift Dzire VDi,2009,150000.0,120000.0,Diesel,Individual,Manual,Second Owner,2438,16,Low,High,0.0,Diesel_Manual,Maruti,1.25,1
2438,CAR_002439,Honda Amaze E i-Dtech,2015,300000.0,104000.0,Electric,Individual,Manual,Second Owner,2439,10,Low,High,0.0,Diesel_Manual,Honda,2.88,1
2439,CAR_002440,Maruti Swift Dzire LXi,2011,150000.0,100000.0,Petrol,Individual,Manual,Second Owner,2440,14,Low,High,0.0,Petrol_Manual,Maruti,1.5,0
2440,CAR_002441,Honda City 1.3 EXI,2003,120000.0,132343.0,Petrol,Individual,Manual,Second Owner,2441,22,Low,High,0.0,Petrol_Manual,Honda,0.91,0
2441,CAR_002442,Hyundai Creta 1.4 CRDi S Plus,2016,950000.0,90000.0,Diesel,Individual,Manual,Second Owner,2442,9,Low,High,0.0,Diesel_Manual,Hyundai,10.56,1
2442,CAR_002443,Maruti Wagon R VXI BS IV,2007,100000.0,60000.0,Petrol,Individual,Manual,Second Owner,2443,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.67,0
2443,CAR_002444,Honda Jazz 1.5 SV i DTEC,2015,450000.0,76000.0,Diesel,Individual,Manual,First Owner,2444,10,Mid,High,0.0,Diesel_Manual,Honda,5.92,1
2444,CAR_002445,Maruti 800 EX,2004,30000.0,60000.0,Petrol,Individual,Manual,Third Owner,2445,21,Low,Medium,0.0,Petrol_Manual,Maruti,0.5,0
2445,CAR_002446,Mahindra Renault Logan 1.6 Petrol GLSX,2009,200000.0,30000.0,Petrol,Individual,Manual,Second Owner,2446,16,Low,High,0.0,Petrol_Manual,Mahindra,6.67,0
2446,CAR_002447,Tata Indigo LS,2010,4461000.0,96000.0,Diesel,Individual,Manual,Second Owner,2447,15,Low,High,0.0,Diesel_Manual,Tata,2.6,1
2447,CAR_002448,Mahindra Marazzo M8 8Str,2018,1300000.0,10000.0,Diesel,Individual,Manual,First Owner,2448,7,Premium,Low,0.0,Diesel_Manual,Mahindra,130.0,1
2448,CAR_002449,Toyota Etios Liva 1.4 VD,2017,425000.0,36000.0,Diesel,Dealer,Manual,First Owner,2449,8,Mid,Medium,0.0,Diesel_Manual,Toyota,11.81,1
2449,CAR_002450,Honda Amaze VX O iDTEC,2017,550000.0,12997.0,Petrol,Dealer,Manual,First Owner,2450,8,Mid,Low,0.0,Diesel_Manual,Honda,42.32,1
2450,CAR_002451,Maruti Ciaz 1.4 AT Zeta,2017,500000.0,40000.0,Petrol,Individual,Automatic,First Owner,2451,8,Mid,Medium,0.0,Petrol_Automatic,Maruti,12.5,0
2451,CAR_002452,Hyundai Verna CRDi 1.6 SX Option,2018,4461000.0,26430.0,Diesel,Dealer,Manual,First Owner,2452,7,Premium,High,0.0,Diesel_Manual,Hyundai,43.51,1
2452,CAR_002453,Maruti Swift 1.3 VXI ABS,2015,475000.0,24600.0,Diesel,Dealer,Manual,First Owner,2453,10,Mid,Low,0.0,Petrol_Manual,Maruti,19.31,0
2453,CAR_002454,Maruti Wagon R LXI,2011,260000.0,28481.0,Petrol,Dealer,Manual,First Owner,2454,14,Low,Low,0.0,Petrol_Manual,Maruti,9.13,0
2454,CAR_002455,Ford Ecosport 1.0 Ecoboost Titanium Optional,2013,350000.0,100000.0,CNG,Individual,Manual,First Owner,2455,12,Mid,High,0.0,Petrol_Manual,Ford,3.5,0
2455,CAR_002456,Mahindra XUV500 W8 2WD,2013,750000.0,41988.0,Diesel,Dealer,Manual,First Owner,2456,12,High,Medium,0.0,Diesel_Manual,Mahindra,17.86,1
2456,CAR_002457,Maruti Wagon R LXI,2009,180000.0,30375.0,Petrol,Dealer,Manual,First Owner,2457,16,Low,Medium,0.0,Petrol_Manual,Maruti,5.93,0
2457,CAR_002458,Renault KWID Climber 1.0 AMT BSIV,2017,350000.0,7658.0,Petrol,Dealer,Automatic,First Owner,2458,8,Mid,Low,0.0,Petrol_Automatic,Renault,45.7,0
2458,CAR_002459,Maruti Swift Dzire VDI,2017,611000.0,34400.0,Diesel,Dealer,Manual,First Owner,2459,8,High,High,0.0,Diesel_Manual,Maruti,17.76,1
2459,CAR_002460,Renault KWID RXL BSIV,2017,4461000.0,18500.0,Petrol,Dealer,Manual,First Owner,2460,8,Mid,Low,0.0,Petrol_Manual,Renault,17.57,0
2460,CAR_002461,Maruti Alto VXi,2015,4461000.0,26134.0,Petrol,Dealer,Manual,First Owner,2461,10,Low,Low,0.0,Petrol_Manual,Maruti,10.71,0
2461,CAR_002462,Maruti Wagon R AMT VXI,2014,300000.0,28942.0,Petrol,Dealer,Automatic,First Owner,2462,11,Low,Low,0.0,Petrol_Automatic,Maruti,10.37,0
2462,CAR_002463,Mahindra XUV500 W8 2WD,2012,711000.0,53600.0,Diesel,Dealer,Manual,First Owner,2463,13,High,Medium,0.0,Diesel_Manual,Mahindra,13.26,1
2463,CAR_002464,Toyota Innova 2.5 G (Diesel) 8 Seater,2015,851000.0,53652.0,Diesel,Dealer,Manual,First Owner,2464,10,High,Medium,0.0,Diesel_Manual,Toyota,15.86,1
2464,CAR_002465,Maruti Ritz VDi,2011,4461000.0,52895.0,Diesel,Dealer,Manual,First Owner,2465,14,Low,Medium,0.0,Diesel_Manual,Maruti,4.92,1
2465,CAR_002466,Hyundai Verna SX Diesel,2013,550000.0,42324.0,Diesel,Dealer,Manual,Second Owner,2466,12,Low,Medium,0.0,Diesel_Manual,Hyundai,12.99,1
2466,CAR_002467,Mahindra XUV500 W10 1.99 mHawk,2016,1044999.0,65000.0,Diesel,Dealer,Manual,First Owner,2467,9,Premium,Medium,0.0,Diesel_Manual,Mahindra,16.08,1
2467,CAR_002468,Mahindra Scorpio EX,2012,550000.0,60236.0,Diesel,Dealer,Manual,First Owner,2468,13,Mid,Medium,0.0,Diesel_Manual,Mahindra,9.13,1
2468,CAR_002469,Maruti Wagon R VXI BS IV,2014,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,2469,11,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
2469,CAR_002470,Maruti Wagon R VXI BS IV,2014,210000.0,70000.0,Petrol,Individual,Manual,Third Owner,2470,11,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
2470,CAR_002471,Mahindra Xylo E8,2009,290000.0,100000.0,Diesel,Individual,Manual,Second Owner,2471,16,Low,High,0.0,Diesel_Manual,Mahindra,2.9,1
2471,CAR_002472,Fiat 500 Lounge,2008,250999.0,110000.0,Diesel,Individual,Manual,First Owner,2472,17,Low,High,0.0,Diesel_Manual,Fiat,2.28,1
2472,CAR_002473,Renault Triber RXT BSIV,2019,600000.0,10300.0,Petrol,Individual,Manual,First Owner,2473,6,Mid,Low,0.0,Petrol_Manual,Renault,58.25,0
2473,CAR_002474,Fiat Grande Punto EVO 1.3 Active,2014,4461000.0,120000.0,LPG,Individual,Manual,Second Owner,2474,11,Low,High,0.0,Diesel_Manual,Fiat,2.25,1
2474,CAR_002475,Tata New Safari 4X4 EX,2012,4461000.0,142000.0,Diesel,Individual,Manual,First Owner,2475,13,Mid,High,0.0,Diesel_Manual,Tata,3.87,1
2475,CAR_002476,Mahindra TUV 300 mHAWK100 T8,2017,800000.0,25000.0,Diesel,Individual,Manual,First Owner,2476,8,High,Low,0.0,Diesel_Manual,Mahindra,32.0,1
2476,CAR_002477,Tata Altroz XE,2020,4461000.0,5000.0,Petrol,Individual,Manual,First Owner,2477,5,Mid,Low,0.0,Petrol_Manual,Tata,100.0,0
2477,CAR_002478,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,2478,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
2478,CAR_002479,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2016,396000.0,28643.0,Petrol,Dealer,Manual,First Owner,2479,9,Mid,Low,0.0,Petrol_Manual,Hyundai,13.83,0
2479,CAR_002480,Honda City i DTEC VX,2014,600000.0,50000.0,Diesel,Individual,Manual,First Owner,2480,11,Mid,Medium,0.0,Diesel_Manual,Honda,12.0,1
2480,CAR_002481,Maruti Wagon R ZXI 1.2,2019,575000.0,30000.0,Petrol,Dealer,Manual,First Owner,2481,6,Mid,Low,0.0,Petrol_Manual,Maruti,19.17,0
2481,CAR_002482,Hyundai Santro Sportz BSIV,2020,520000.0,5000.0,Petrol,Individual,Manual,First Owner,2482,5,Mid,Low,0.0,Petrol_Manual,Hyundai,104.0,0
2482,CAR_002483,Mahindra XUV500 AT W10 FWD,2018,1400000.0,30000.0,Diesel,Individual,Automatic,First Owner,2483,7,Low,Low,0.0,Diesel_Automatic,Mahindra,46.67,1
2483,CAR_002484,Maruti Baleno Delta Automatic,2018,600000.0,7600.0,Electric,Individual,Automatic,First Owner,2484,7,Mid,Low,0.0,Petrol_Automatic,Maruti,78.95,0
2484,CAR_002485,Chevrolet Beat Diesel LS,2013,195000.0,60000.0,Petrol,Dealer,Manual,First Owner,2485,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,4.13,1
2485,CAR_002486,Maruti Alto K10 VXI AGS,2015,281000.0,60000.0,Petrol,Dealer,Automatic,Second Owner,2486,10,Low,Low,0.0,Petrol_Automatic,Maruti,63.4,0
2486,CAR_002487,Tata Indica GLS BS IV,2010,75000.0,110000.0,Petrol,Individual,Manual,Second Owner,2487,15,Low,High,0.0,Petrol_Manual,Tata,0.68,0
2487,CAR_002488,Hyundai Grand i10 Sportz,2014,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,2488,11,Mid,Low,0.0,Petrol_Manual,Hyundai,16.0,0
2488,CAR_002489,Tata Indica Vista Terra 1.4 TDI,2009,93000.0,70000.0,Diesel,Individual,Manual,Second Owner,2489,16,Low,Medium,0.0,Diesel_Manual,Tata,1.33,1
2489,CAR_002490,Hyundai i20 Active S Petrol,2016,560000.0,60000.0,Petrol,Dealer,Manual,First Owner,2490,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.17,0
2490,CAR_002491,Hyundai EON 1.0 Era Plus,2019,400000.0,5000.0,Petrol,Individual,Manual,First Owner,2491,6,Mid,Low,0.0,Petrol_Manual,Hyundai,80.0,0
2491,CAR_002492,Tata Tigor 1.2 Revotron XT,2018,459999.0,40000.0,Petrol,Individual,Manual,First Owner,2492,7,Mid,Medium,0.0,Petrol_Manual,Tata,11.5,0
2492,CAR_002493,Hyundai Xcent 1.2 CRDi E,2017,450000.0,43000.0,Diesel,Individual,Manual,First Owner,2493,8,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.47,1
2493,CAR_002494,Maruti Alto Green LXi (CNG),2010,4461000.0,80251.0,Diesel,Dealer,Manual,Second Owner,2494,15,Low,High,0.0,CNG_Manual,Maruti,1.1,0
2494,CAR_002495,Hyundai i20 1.2 Era,2010,4461000.0,34500.0,Petrol,Individual,Manual,Second Owner,2495,15,Low,Medium,0.0,Petrol_Manual,Hyundai,7.25,0
2495,CAR_002496,Ford Ikon 1.4 ZXi,2000,22000.0,42743.0,CNG,Dealer,Manual,Third Owner,2496,25,Low,High,0.0,Petrol_Manual,Ford,0.51,0
2496,CAR_002497,Chevrolet Beat Diesel LS,2013,130000.0,60000.0,Diesel,Individual,Manual,Second Owner,2497,12,Low,High,0.0,Diesel_Manual,Chevrolet,1.38,1
2497,CAR_002498,Maruti Zen VXi - BS III,2004,79000.0,80000.0,Petrol,Individual,Manual,First Owner,2498,21,Low,High,0.0,Petrol_Manual,Maruti,0.99,0
2498,CAR_002499,Chevrolet Beat Diesel LS,2012,120000.0,60000.0,Diesel,Individual,Manual,First Owner,2499,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.09,1
2499,CAR_002500,BMW 7 Series 730Ld,2011,1700000.0,100000.0,Diesel,Individual,Automatic,Second Owner,2500,14,Premium,High,0.0,Diesel_Automatic,BMW,17.0,1
2500,CAR_002501,Chevrolet Optra Magnum 2.0 LT,2012,300000.0,25000.0,Diesel,Individual,Manual,First Owner,2501,13,Low,Low,0.0,Diesel_Manual,Chevrolet,12.0,1
2501,CAR_002502,Chevrolet Optra Magnum 2.0 LT,2012,300000.0,25000.0,Diesel,Individual,Manual,First Owner,2502,13,Low,Low,0.0,Diesel_Manual,Chevrolet,12.0,1
2502,CAR_002503,Maruti A-Star Lxi,2009,145000.0,25000.0,Petrol,Individual,Manual,First Owner,2503,16,Low,Low,0.0,Petrol_Manual,Maruti,5.8,0
2503,CAR_002504,Ford Endeavour 3.2 Titanium AT 4X4,2019,3200000.0,20000.0,Diesel,Individual,Automatic,First Owner,2504,6,Premium,Low,0.0,Diesel_Automatic,Ford,160.0,1
2504,CAR_002505,Chevrolet Spark 1.0 LT,2013,240000.0,10000.0,Petrol,Individual,Manual,First Owner,2505,12,Low,Low,0.0,Petrol_Manual,Chevrolet,24.0,0
2505,CAR_002506,Honda WR-V i-DTEC VX,2018,4461000.0,30000.0,Diesel,Individual,Manual,First Owner,2506,7,High,Low,0.0,Diesel_Manual,Honda,23.33,1
2506,CAR_002507,Hyundai Accent GLX,2010,145000.0,110000.0,Petrol,Individual,Manual,Second Owner,2507,15,Low,High,0.0,Petrol_Manual,Hyundai,1.32,0
2507,CAR_002508,Maruti Ertiga VDI Limited Edition,2014,535000.0,80000.0,Diesel,Individual,Manual,Second Owner,2508,11,Mid,High,0.0,Diesel_Manual,Maruti,6.69,1
2508,CAR_002509,Mahindra KUV 100 mFALCON G80 K2 Plus,2016,420000.0,21000.0,LPG,Individual,Manual,First Owner,2509,9,Mid,High,0.0,Petrol_Manual,Mahindra,20.0,0
2509,CAR_002510,Maruti Alto K10 VXI,2016,350000.0,20000.0,Petrol,Individual,Manual,First Owner,2510,9,Mid,Low,0.0,Petrol_Manual,Maruti,17.5,0
2510,CAR_002511,Maruti Swift VDI BSIV,2015,550000.0,32000.0,Diesel,Individual,Manual,First Owner,2511,10,Low,Medium,0.0,Diesel_Manual,Maruti,17.19,1
2511,CAR_002512,Mitsubishi Montero 3.2 MT,2007,750000.0,180000.0,Diesel,Individual,Manual,First Owner,2512,18,High,Very High,0.0,Diesel_Manual,Mitsubishi,4.17,1
2512,CAR_002513,Tata Indica Vista Aqua 1.3 Quadrajet ABS BSIV,2012,4461000.0,70000.0,Diesel,Individual,Manual,Second Owner,2513,13,Low,Medium,0.0,Diesel_Manual,Tata,2.83,1
2513,CAR_002514,Hyundai i20 Magna Optional 1.4 CRDi,2013,420000.0,50000.0,Diesel,Individual,Manual,First Owner,2514,12,Mid,Medium,0.0,Diesel_Manual,Hyundai,8.4,1
2514,CAR_002515,Hyundai i20 Magna Optional 1.4 CRDi,2013,420000.0,50000.0,Diesel,Individual,Manual,First Owner,2515,12,Mid,Medium,0.0,Diesel_Manual,Hyundai,8.4,1
2515,CAR_002516,Ford Endeavour 4x2 XLT,2004,195000.0,120000.0,Diesel,Individual,Manual,First Owner,2516,21,Low,High,0.0,Diesel_Manual,Ford,1.62,1
2516,CAR_002517,Hyundai EON Sportz,2012,150000.0,55766.0,Petrol,Individual,Manual,Second Owner,2517,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.69,0
2517,CAR_002518,Nissan Micra XL,2011,275000.0,40000.0,Electric,Individual,Manual,Second Owner,2518,14,Low,Medium,0.0,Petrol_Manual,Nissan,6.88,0
2518,CAR_002519,Maruti 800 Std,2001,65000.0,60000.0,Petrol,Individual,Manual,Third Owner,2519,24,Low,Medium,0.0,Petrol_Manual,Maruti,1.08,0
2519,CAR_002520,Maruti Swift LXi BSIV,2008,200000.0,50000.0,Petrol,Individual,Manual,First Owner,2520,17,Low,Medium,0.0,Petrol_Manual,Maruti,4.0,0
2520,CAR_002521,Maruti Swift Dzire VDi,2010,250000.0,110000.0,Diesel,Individual,Manual,Second Owner,2521,15,Low,High,0.0,Diesel_Manual,Maruti,2.27,1
2521,CAR_002522,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,2522,12,Mid,Low,0.0,Diesel_Manual,Renault,450.0,1
2522,CAR_002523,Maruti Ignis 1.3 Delta,2018,590000.0,26350.0,Diesel,Dealer,Manual,First Owner,2523,7,Mid,Low,0.0,Diesel_Manual,Maruti,22.39,1
2523,CAR_002524,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,2524,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
2524,CAR_002525,Tata Indigo CS LX (TDI) BS-III,2015,225000.0,68745.0,Diesel,Dealer,Manual,First Owner,2525,10,Low,Medium,0.0,Diesel_Manual,Tata,3.27,1
2525,CAR_002526,Maruti Alto 800 LXI,2006,165000.0,132000.0,Petrol,Individual,Manual,First Owner,2526,19,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
2526,CAR_002527,Mahindra Bolero Power Plus SLX,2019,860000.0,35000.0,Diesel,Individual,Manual,First Owner,2527,6,High,Medium,0.0,Diesel_Manual,Mahindra,24.57,1
2527,CAR_002528,Mahindra XUV500 W6 2WD,2016,900000.0,47000.0,Diesel,Individual,Manual,First Owner,2528,9,High,Medium,0.0,Diesel_Manual,Mahindra,19.15,1
2528,CAR_002529,Tata Indigo TDI,2013,182000.0,100000.0,Diesel,Individual,Manual,First Owner,2529,12,Low,High,0.0,Diesel_Manual,Tata,1.82,1
2529,CAR_002530,Tata Indica Vista Terra 1.4 TDI,2010,100000.0,80000.0,Diesel,Individual,Manual,First Owner,2530,15,Low,High,0.0,Diesel_Manual,Tata,1.25,1
2530,CAR_002531,Hyundai Santro Xing XL eRLX Euro III,2006,100000.0,40000.0,Petrol,Individual,Manual,Second Owner,2531,19,Low,Medium,0.0,Petrol_Manual,Hyundai,2.5,0
2531,CAR_002532,Maruti Alto K10 VXI AGS,2017,375000.0,27289.0,Petrol,Dealer,Automatic,First Owner,2532,8,Mid,Low,0.0,Petrol_Automatic,Maruti,13.74,0
2532,CAR_002533,Tata Tiago XZA AMT,2018,525000.0,10980.0,Petrol,Dealer,Automatic,First Owner,2533,7,Mid,Low,0.0,Petrol_Automatic,Tata,47.81,0
2533,CAR_002534,Hyundai EON Era Plus,2016,320000.0,24662.0,Petrol,Dealer,Manual,First Owner,2534,9,Mid,Low,0.0,Petrol_Manual,Hyundai,12.98,0
2534,CAR_002535,Maruti Swift Dzire VDI,2013,525000.0,37000.0,Diesel,Dealer,Manual,First Owner,2535,12,Mid,Medium,0.0,Diesel_Manual,Maruti,14.19,1
2535,CAR_002536,Nissan Evalia XV,2014,650000.0,28245.0,Diesel,Dealer,Manual,First Owner,2536,11,High,Low,0.0,Diesel_Manual,Nissan,23.01,1
2536,CAR_002537,Hyundai Grand i10 1.2 CRDi Sportz Option,2017,4461000.0,27005.0,Petrol,Dealer,Manual,First Owner,2537,8,Mid,Low,0.0,Diesel_Manual,Hyundai,21.29,1
2537,CAR_002538,Hyundai i10 Magna,2014,355000.0,39227.0,Petrol,Dealer,Manual,First Owner,2538,11,Low,High,0.0,Petrol_Manual,Hyundai,9.05,0
2538,CAR_002539,Mahindra KUV 100 D75 K4 Plus 5Str,2016,470000.0,31367.0,Diesel,Dealer,Manual,First Owner,2539,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,14.98,1
2539,CAR_002540,Maruti Wagon R LXI,2008,221000.0,35008.0,Diesel,Dealer,Manual,First Owner,2540,17,Low,High,0.0,Petrol_Manual,Maruti,6.31,0
2540,CAR_002541,Hyundai Santro Xing GLS,2009,195000.0,27005.0,CNG,Dealer,Manual,First Owner,2541,16,Low,Low,0.0,Petrol_Manual,Hyundai,7.22,0
2541,CAR_002542,Maruti Zen VX,2001,85000.0,86000.0,Petrol,Individual,Manual,Second Owner,2542,24,Low,High,0.0,Petrol_Manual,Maruti,0.99,0
2542,CAR_002543,Maruti Swift VDI,2013,400000.0,50000.0,Diesel,Individual,Manual,First Owner,2543,12,Mid,High,0.0,Diesel_Manual,Maruti,8.0,1
2543,CAR_002544,Maruti Baleno Delta 1.3,2017,650000.0,20000.0,Diesel,Individual,Manual,First Owner,2544,8,High,Low,0.0,Diesel_Manual,Maruti,32.5,1
2544,CAR_002545,Hyundai i10 Magna 1.2,2010,400000.0,50000.0,Petrol,Individual,Manual,First Owner,2545,15,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.0,0
2545,CAR_002546,Tata Manza Aura (ABS) Quadrajet BS IV,2012,295000.0,80000.0,Diesel,Individual,Manual,Second Owner,2546,13,Low,High,0.0,Diesel_Manual,Tata,3.69,1
2546,CAR_002547,Maruti Omni LPG STD BSIV,2006,60000.0,135000.0,LPG,Individual,Manual,Fourth & Above Owner,2547,19,Low,High,0.0,LPG_Manual,Maruti,0.44,0
2547,CAR_002548,Maruti Zen D PS,2003,4461000.0,120000.0,Diesel,Individual,Manual,Third Owner,2548,22,Low,High,0.0,Diesel_Manual,Maruti,0.92,1
2548,CAR_002549,Fiat Linea 1.3 Multijet Emotion,2015,4461000.0,113600.0,Diesel,Individual,Manual,First Owner,2549,10,Mid,High,0.0,Diesel_Manual,Fiat,4.84,1
2549,CAR_002550,Hyundai Santro Xing GLS,2008,190000.0,60000.0,Petrol,Individual,Manual,Third Owner,2550,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.71,0
2550,CAR_002551,Ford Aspire Titanium Plus Diesel BSIV,2018,861999.0,20000.0,Diesel,Dealer,Manual,First Owner,2551,7,High,Low,0.0,Diesel_Manual,Ford,43.1,1
2551,CAR_002552,Ford Aspire Titanium Plus BSIV,2018,782000.0,14000.0,Petrol,Dealer,Manual,First Owner,2552,7,High,Low,0.0,Petrol_Manual,Ford,55.86,0
2552,CAR_002553,Ford Freestyle Titanium Plus Diesel BSIV,2018,836000.0,26000.0,Diesel,Dealer,Manual,First Owner,2553,7,High,Low,0.0,Diesel_Manual,Ford,32.15,1
2553,CAR_002554,Honda City i VTEC V,2016,696000.0,138925.0,LPG,Dealer,Manual,First Owner,2554,9,High,High,0.0,Petrol_Manual,Honda,5.01,0
2554,CAR_002555,Honda Jazz 1.5 VX i DTEC,2016,4461000.0,121764.0,Diesel,Dealer,Manual,First Owner,2555,9,Mid,High,0.0,Diesel_Manual,Honda,4.89,1
2555,CAR_002556,Honda Jazz 1.2 VX i VTEC,2016,612000.0,60000.0,Petrol,Dealer,Manual,First Owner,2556,9,High,High,0.0,Petrol_Manual,Honda,5.8,0
2556,CAR_002557,Maruti Alto 800 LXI,2015,114999.0,35000.0,Petrol,Individual,Manual,First Owner,2557,10,Low,Medium,0.0,Petrol_Manual,Maruti,3.29,0
2557,CAR_002558,Tata Bolt Quadrajet XE,2017,340000.0,60000.0,Diesel,Individual,Manual,First Owner,2558,8,Mid,Medium,0.0,Diesel_Manual,Tata,5.67,1
2558,CAR_002559,Maruti Swift ZXI Plus,2020,550000.0,5000.0,Petrol,Individual,Manual,First Owner,2559,5,Mid,Low,0.0,Petrol_Manual,Maruti,110.0,0
2559,CAR_002560,Chevrolet Aveo U-VA 1.2 LT,2008,114999.0,70000.0,Petrol,Individual,Manual,Third Owner,2560,17,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.64,0
2560,CAR_002561,Maruti Wagon R VXI,2000,105000.0,60000.0,Petrol,Individual,Manual,First Owner,2561,25,Low,High,0.0,Petrol_Manual,Maruti,1.31,0
2561,CAR_002562,Maruti Vitara Brezza ZDi,2019,890000.0,20000.0,Diesel,Individual,Manual,First Owner,2562,6,High,Low,0.0,Diesel_Manual,Maruti,44.5,1
2562,CAR_002563,Tata Nexon 1.2 Revotron XZ Plus,2019,715000.0,8000.0,Petrol,Individual,Manual,First Owner,2563,6,High,Low,0.0,Petrol_Manual,Tata,89.38,0
2563,CAR_002564,Hyundai i20 Asta (o),2014,525000.0,54000.0,Electric,Dealer,Manual,Second Owner,2564,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.72,0
2564,CAR_002565,Volkswagen Jetta 2.0L TDI Highline AT,2012,735000.0,55300.0,Diesel,Dealer,Manual,First Owner,2565,13,High,Medium,0.0,Diesel_Automatic,Volkswagen,13.29,1
2565,CAR_002566,Honda Amaze S i-VTEC,2016,495000.0,11958.0,Petrol,Dealer,Manual,First Owner,2566,9,Mid,Low,0.0,Petrol_Manual,Honda,41.39,0
2566,CAR_002567,Hyundai Creta 1.6 SX Option,2017,1025000.0,9000.0,Petrol,Dealer,Manual,First Owner,2567,8,Premium,Low,0.0,Petrol_Manual,Hyundai,113.89,0
2567,CAR_002568,Mahindra XUV500 W8 2WD,2015,750000.0,60000.0,Diesel,Individual,Manual,First Owner,2568,10,High,Medium,0.0,Diesel_Manual,Mahindra,10.71,1
2568,CAR_002569,Renault Duster 85PS Diesel RxL Optional,2013,600000.0,120000.0,Diesel,Individual,Manual,First Owner,2569,12,Mid,High,0.0,Diesel_Manual,Renault,5.0,1
2569,CAR_002570,Hyundai i20 Asta 1.2,2014,475000.0,23122.0,Petrol,Dealer,Manual,Second Owner,2570,11,Mid,Low,0.0,Petrol_Manual,Hyundai,20.54,0
2570,CAR_002571,Hyundai Santro Xing XO,2007,80000.0,58000.0,Petrol,Dealer,Manual,Second Owner,2571,18,Low,Medium,0.0,Petrol_Manual,Hyundai,1.38,0
2571,CAR_002572,Mahindra Bolero 2011-2019 SLE,2013,325000.0,62200.0,Diesel,Dealer,Manual,First Owner,2572,12,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.23,1
2572,CAR_002573,Audi A6 2.0 TDI Premium Plus,2014,4461000.0,34000.0,Diesel,Dealer,Automatic,Second Owner,2573,11,Premium,Medium,0.0,Diesel_Automatic,Audi,43.24,1
2573,CAR_002574,Fiat Avventura MULTIJET Emotion,2015,350000.0,53000.0,Diesel,Individual,Manual,Second Owner,2574,10,Mid,Medium,0.0,Diesel_Manual,Fiat,6.6,1
2574,CAR_002575,Audi A8 4.2 TDI,2013,2800000.0,60000.0,Diesel,Dealer,Manual,First Owner,2575,12,Premium,Medium,0.0,Diesel_Automatic,Audi,57.14,1
2575,CAR_002576,Datsun RediGO 1.0 S,2017,210000.0,15000.0,Petrol,Dealer,Manual,Second Owner,2576,8,Low,Low,0.0,Petrol_Manual,Datsun,14.0,0
2576,CAR_002577,Chevrolet Cruze LT,2012,349000.0,44500.0,CNG,Dealer,Manual,Second Owner,2577,13,Mid,Medium,0.0,Diesel_Manual,Chevrolet,7.84,1
2577,CAR_002578,Volkswagen Jetta 1.4 TSI Comfortline,2013,450000.0,50000.0,Petrol,Individual,Manual,First Owner,2578,12,Mid,High,0.0,Petrol_Manual,Volkswagen,9.0,0
2578,CAR_002579,Audi A4 2.0 TDI 177 Bhp Premium Plus,2013,1150000.0,53000.0,Diesel,Dealer,Automatic,First Owner,2579,12,Premium,Medium,0.0,Diesel_Automatic,Audi,21.7,1
2579,CAR_002580,Honda Civic 1.8 V AT,2009,210000.0,63500.0,Petrol,Dealer,Automatic,First Owner,2580,16,Low,Medium,0.0,Petrol_Automatic,Honda,3.31,0
2580,CAR_002581,Mercedes-Benz E-Class Exclusive E 200 BSIV,2018,4500000.0,9800.0,Petrol,Dealer,Automatic,First Owner,2581,7,Premium,High,0.0,Petrol_Automatic,Mercedes-Benz,459.18,0
2581,CAR_002582,Volkswagen Polo GTI,2017,825000.0,13599.0,Petrol,Dealer,Automatic,First Owner,2582,8,High,Low,0.0,Petrol_Automatic,Volkswagen,60.67,0
2582,CAR_002583,BMW X1 sDrive 20d xLine,2017,2750000.0,13000.0,Diesel,Individual,Automatic,First Owner,2583,8,Premium,Low,0.0,Diesel_Automatic,BMW,211.54,1
2583,CAR_002584,Chevrolet Beat Diesel,2012,160000.0,110000.0,Diesel,Individual,Manual,First Owner,2584,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.45,1
2584,CAR_002585,Mahindra Scorpio SLE BSIV,2014,550000.0,100000.0,Diesel,Individual,Manual,Second Owner,2585,11,Mid,High,0.0,Diesel_Manual,Mahindra,5.5,1
2585,CAR_002586,Maruti Swift 1.3 VXI ABS,2006,95000.0,110000.0,Petrol,Individual,Manual,Third Owner,2586,19,Low,High,0.0,Petrol_Manual,Maruti,0.86,0
2586,CAR_002587,Toyota Innova 2.5 G (Diesel) 7 Seater,2014,725000.0,90000.0,Diesel,Individual,Manual,Second Owner,2587,11,High,High,0.0,Diesel_Manual,Toyota,8.06,1
2587,CAR_002588,Tata Tigor 1.2 Revotron XZ Option,2019,670000.0,17000.0,Petrol,Individual,Manual,First Owner,2588,6,High,High,0.0,Petrol_Manual,Tata,39.41,0
2588,CAR_002589,Maruti Alto LX,2008,65000.0,140000.0,Petrol,Individual,Manual,First Owner,2589,17,Low,High,0.0,Petrol_Manual,Maruti,0.46,0
2589,CAR_002590,Mahindra Xylo D4,2015,4461000.0,50000.0,Diesel,Individual,Manual,Second Owner,2590,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,10.0,1
2590,CAR_002591,Mahindra KUV 100 mFALCON G80 K8 5str,2017,600000.0,53000.0,Petrol,Individual,Manual,First Owner,2591,8,Mid,Medium,0.0,Petrol_Manual,Mahindra,11.32,0
2591,CAR_002592,Maruti Wagon R VXI Optional,2017,350000.0,70000.0,Petrol,Individual,Manual,Second Owner,2592,8,Mid,Medium,0.0,Petrol_Manual,Maruti,5.0,0
2592,CAR_002593,Chevrolet Beat Diesel LT,2012,200000.0,70000.0,Diesel,Individual,Manual,Second Owner,2593,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.86,1
2593,CAR_002594,Hyundai Getz 1.5 CRDi GVS,2008,110000.0,154000.0,Diesel,Individual,Manual,Third Owner,2594,17,Low,Very High,0.0,Diesel_Manual,Hyundai,0.71,1
2594,CAR_002595,Hyundai EON Sportz,2012,150000.0,80000.0,Petrol,Individual,Manual,First Owner,2595,13,Low,High,0.0,Petrol_Manual,Hyundai,1.88,0
2595,CAR_002596,Maruti Zen D PS,2003,75000.0,110000.0,Diesel,Individual,Manual,Second Owner,2596,22,Low,High,0.0,Diesel_Manual,Maruti,0.68,1
2596,CAR_002597,Honda Amaze EX i-Dtech,2013,4461000.0,90000.0,Diesel,Individual,Manual,Second Owner,2597,12,Mid,High,0.0,Diesel_Manual,Honda,3.89,1
2597,CAR_002598,Toyota Innova Crysta 2.4 VX MT 8S BSIV,2018,1700000.0,60000.0,Diesel,Individual,Manual,First Owner,2598,7,Premium,Low,0.0,Diesel_Manual,Toyota,113.33,1
2598,CAR_002599,BMW X1 sDrive20d,2011,890000.0,86000.0,Diesel,Individual,Automatic,Second Owner,2599,14,High,High,0.0,Diesel_Automatic,BMW,10.35,1
2599,CAR_002600,Mahindra Scorpio M2DI,2014,500000.0,120000.0,Diesel,Individual,Manual,First Owner,2600,11,Mid,High,0.0,Diesel_Manual,Mahindra,4.17,1
2600,CAR_002601,Hyundai Elite i20 Sportz Plus Dual Tone BSIV,2019,600000.0,5200.0,Petrol,Individual,Manual,First Owner,2601,6,Mid,High,0.0,Petrol_Manual,Hyundai,115.38,0
2601,CAR_002602,Chevrolet Beat LS,2010,99000.0,60000.0,Petrol,Individual,Manual,Second Owner,2602,15,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.65,0
2602,CAR_002603,Hyundai i10 Era 1.1 iTech SE,2011,235000.0,46000.0,Petrol,Individual,Manual,First Owner,2603,14,Low,Medium,0.0,Petrol_Manual,Hyundai,5.11,0
2603,CAR_002604,Tata Indigo GLX,2006,75000.0,40000.0,Petrol,Individual,Manual,Second Owner,2604,19,Low,Medium,0.0,Petrol_Manual,Tata,1.88,0
2604,CAR_002605,Honda City i DTEC VX,2014,600000.0,90000.0,LPG,Individual,Manual,Second Owner,2605,11,Mid,High,0.0,Diesel_Manual,Honda,6.67,1
2605,CAR_002606,Hyundai Verna CRDi 1.6 SX,2019,4461000.0,25000.0,Diesel,Individual,Manual,Second Owner,2606,6,High,Low,0.0,Diesel_Manual,Hyundai,40.0,1
2606,CAR_002607,Hyundai Verna 1.6 SX,2012,390000.0,60000.0,Diesel,Individual,Manual,First Owner,2607,13,Mid,High,0.0,Diesel_Manual,Hyundai,4.33,1
2607,CAR_002608,Volkswagen Polo 1.5 TDI Highline,2015,600000.0,70000.0,Diesel,Individual,Manual,First Owner,2608,10,Mid,Medium,0.0,Diesel_Manual,Volkswagen,8.57,1
2608,CAR_002609,Hyundai i20 1.2 Spotz,2017,600000.0,70000.0,Petrol,Individual,Manual,Second Owner,2609,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.57,0
2609,CAR_002610,Hyundai Elite i20 Magna Plus Diesel,2019,730000.0,35000.0,Diesel,Individual,Manual,Second Owner,2610,6,High,Medium,0.0,Diesel_Manual,Hyundai,20.86,1
2610,CAR_002611,Hyundai Grand i10 1.2 Kappa Magna AT,2017,450000.0,12700.0,Petrol,Individual,Automatic,First Owner,2611,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,35.43,0
2611,CAR_002612,Maruti Ritz LDi,2011,250000.0,80000.0,Diesel,Individual,Manual,First Owner,2612,14,Low,High,0.0,Diesel_Manual,Maruti,3.12,1
2612,CAR_002613,Mahindra XUV500 W10 2WD,2018,1400000.0,20000.0,Diesel,Individual,Manual,First Owner,2613,7,Premium,Low,0.0,Diesel_Manual,Mahindra,70.0,1
2613,CAR_002614,Renault KWID 1.0 RXT Optional,2018,300000.0,20000.0,Petrol,Individual,Manual,First Owner,2614,7,Low,Low,0.0,Petrol_Manual,Renault,15.0,0
2614,CAR_002615,Ford EcoSport 1.5 Petrol Titanium BSIV,2017,800000.0,19000.0,Petrol,Individual,Manual,First Owner,2615,8,High,Low,0.0,Petrol_Manual,Ford,42.11,0
2615,CAR_002616,Fiat Grande Punto Active (Diesel),2009,125000.0,95000.0,Diesel,Individual,Manual,Second Owner,2616,16,Low,High,0.0,Diesel_Manual,Fiat,1.32,1
2616,CAR_002617,BMW X1 sDrive 20d xLine,2019,2600000.0,9500.0,Diesel,Individual,Automatic,First Owner,2617,6,Premium,Low,0.0,Diesel_Automatic,BMW,273.68,1
2617,CAR_002618,Skoda Fabia 1.2 MPI Ambition Plus,2012,300000.0,50000.0,Petrol,Individual,Manual,First Owner,2618,13,Low,Medium,0.0,Petrol_Manual,Skoda,6.0,0
2618,CAR_002619,Volkswagen Polo Diesel Trendline 1.2L,2012,295000.0,90000.0,Electric,Individual,Manual,First Owner,2619,13,Low,High,0.0,Diesel_Manual,Volkswagen,3.28,1
2619,CAR_002620,Ford Ecosport 1.0 Ecoboost Platinum Edition BSIV,2013,4461000.0,45839.0,Petrol,Dealer,Manual,First Owner,2620,12,High,Medium,0.0,Petrol_Manual,Ford,14.18,0
2620,CAR_002621,Chevrolet Aveo 1.4 LS,2010,4461000.0,74510.0,Petrol,Dealer,Manual,First Owner,2621,15,Low,High,0.0,Petrol_Manual,Chevrolet,2.68,0
2621,CAR_002622,Chevrolet Aveo U-VA 1.2 LS,2009,130000.0,87293.0,Petrol,Dealer,Manual,First Owner,2622,16,Low,High,0.0,Petrol_Manual,Chevrolet,1.49,0
2622,CAR_002623,Ford Figo Diesel EXI,2012,250000.0,156040.0,Diesel,Dealer,Manual,First Owner,2623,13,Low,Very High,0.0,Diesel_Manual,Ford,1.6,1
2623,CAR_002624,Ford Endeavour 2.5L 4X2 MT,2004,400000.0,93415.0,Diesel,Dealer,Manual,First Owner,2624,21,Mid,High,0.0,Diesel_Manual,Ford,4.28,1
2624,CAR_002625,Ford Aspire Titanium BSIV,2017,430000.0,101159.0,Petrol,Dealer,Manual,First Owner,2625,8,Mid,High,0.0,Petrol_Manual,Ford,4.25,0
2625,CAR_002626,Hyundai EON 1.0 Kappa Magna Plus,2015,170000.0,60000.0,Petrol,Individual,Manual,Second Owner,2626,10,Low,High,0.0,Petrol_Manual,Hyundai,2.12,0
2626,CAR_002627,Tata Tiago 2019-2020 XZ,2019,450000.0,60000.0,Petrol,Individual,Manual,First Owner,2627,6,Mid,Low,0.0,Petrol_Manual,Tata,37.5,0
2627,CAR_002628,Maruti Swift ZXI,2013,350000.0,90000.0,Petrol,Individual,Manual,First Owner,2628,12,Mid,High,0.0,Petrol_Manual,Maruti,3.89,0
2628,CAR_002629,Chevrolet Spark 1.0 LT,2011,4461000.0,68519.0,Petrol,Dealer,Manual,First Owner,2629,14,Low,High,0.0,Petrol_Manual,Chevrolet,1.9,0
2629,CAR_002630,Chevrolet Beat Diesel LS,2012,180000.0,55130.0,Petrol,Dealer,Manual,First Owner,2630,13,Low,High,0.0,Diesel_Manual,Chevrolet,3.27,1
2630,CAR_002631,Ford Ecosport 1.0 Ecoboost Titanium Optional,2014,475000.0,65239.0,Petrol,Dealer,Manual,First Owner,2631,11,Mid,Medium,0.0,Petrol_Manual,Ford,7.28,0
2631,CAR_002632,Volkswagen Polo 1.2 MPI Comfortline,2015,470000.0,58182.0,Petrol,Dealer,Manual,First Owner,2632,10,Mid,Medium,0.0,Petrol_Manual,Volkswagen,8.08,0
2632,CAR_002633,Ford Fiesta Diesel Trend,2012,200000.0,91245.0,Diesel,Dealer,Manual,First Owner,2633,13,Low,High,0.0,Diesel_Manual,Ford,2.19,1
2633,CAR_002634,Mahindra XUV500 W8 2WD,2013,650000.0,102989.0,Diesel,Dealer,Manual,First Owner,2634,12,High,High,0.0,Diesel_Manual,Mahindra,6.31,1
2634,CAR_002635,Honda City 1.5 E MT,2010,400000.0,50000.0,Petrol,Individual,Manual,First Owner,2635,15,Mid,Medium,0.0,Petrol_Manual,Honda,8.0,0
2635,CAR_002636,Maruti Zen VXI,2000,100000.0,120000.0,Petrol,Individual,Manual,First Owner,2636,25,Low,High,0.0,Petrol_Manual,Maruti,0.83,0
2636,CAR_002637,Maruti Swift Dzire VDI,2014,409999.0,90000.0,Diesel,Individual,Manual,First Owner,2637,11,Mid,High,0.0,Diesel_Manual,Maruti,4.56,1
2637,CAR_002638,Maruti Swift VDI BSIV,2018,640000.0,25000.0,Diesel,Individual,Manual,First Owner,2638,7,Low,Low,0.0,Diesel_Manual,Maruti,25.6,1
2638,CAR_002639,Maruti Wagon R LXI Minor,2007,4461000.0,80000.0,Petrol,Individual,Manual,Second Owner,2639,18,Low,High,0.0,Petrol_Manual,Maruti,2.0,0
2639,CAR_002640,Tata Indigo TDI,2013,182000.0,100000.0,Diesel,Individual,Manual,First Owner,2640,12,Low,High,0.0,Diesel_Manual,Tata,1.82,1
2640,CAR_002641,Maruti Baleno Zeta 1.3,2018,750000.0,70000.0,CNG,Individual,Manual,First Owner,2641,7,High,Medium,0.0,Diesel_Manual,Maruti,10.71,1
2641,CAR_002642,Mahindra Jeep MM 550 XDB,2002,150000.0,50000.0,Diesel,Individual,Manual,Third Owner,2642,23,Low,Medium,0.0,Diesel_Manual,Mahindra,3.0,1
2642,CAR_002643,Maruti Alto STD,2008,80000.0,100000.0,Petrol,Individual,Manual,Second Owner,2643,17,Low,High,0.0,Petrol_Manual,Maruti,0.8,0
2643,CAR_002644,Maruti Celerio VDi,2016,240000.0,60000.0,Diesel,Individual,Manual,Third Owner,2644,9,Low,High,0.0,Diesel_Manual,Maruti,2.67,1
2644,CAR_002645,Maruti Esteem Lxi,2005,60000.0,60000.0,Petrol,Individual,Manual,Third Owner,2645,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
2645,CAR_002646,Maruti Alto LX,2007,70000.0,120000.0,LPG,Individual,Manual,Third Owner,2646,18,Low,High,0.0,Petrol_Manual,Maruti,0.58,0
2646,CAR_002647,Volkswagen Jetta 1.9 Highline TDI,2008,285000.0,90000.0,Diesel,Individual,Automatic,Second Owner,2647,17,Low,High,0.0,Diesel_Automatic,Volkswagen,3.17,1
2647,CAR_002648,Maruti Swift VDI BSIV,2015,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,2648,10,Mid,Medium,0.0,Diesel_Manual,Maruti,5.83,1
2648,CAR_002649,Hyundai i20 Active 1.4 SX with AVN,2015,699000.0,30000.0,Diesel,Individual,Manual,Second Owner,2649,10,Low,High,0.0,Diesel_Manual,Hyundai,23.3,1
2649,CAR_002650,Ford Figo Diesel ZXI,2011,275000.0,108000.0,Diesel,Individual,Manual,Second Owner,2650,14,Low,High,0.0,Diesel_Manual,Ford,2.55,1
2650,CAR_002651,Hyundai i20 1.4 CRDi Magna,2012,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,2651,13,Mid,High,0.0,Diesel_Manual,Hyundai,2.19,1
2651,CAR_002652,Maruti Alto LXi BSIII,2006,135000.0,90000.0,Petrol,Individual,Manual,Third Owner,2652,19,Low,High,0.0,Petrol_Manual,Maruti,1.5,0
2652,CAR_002653,Mahindra Xylo E4 8S,2010,325000.0,120000.0,Diesel,Individual,Manual,Third Owner,2653,15,Mid,High,0.0,Diesel_Manual,Mahindra,2.71,1
2653,CAR_002654,Mahindra XUV500 W8 2WD,2012,650000.0,102000.0,Diesel,Individual,Manual,Second Owner,2654,13,High,High,0.0,Diesel_Manual,Mahindra,6.37,1
2654,CAR_002655,Mahindra Bolero SLE,2009,300000.0,178000.0,Diesel,Individual,Manual,Second Owner,2655,16,Low,Very High,0.0,Diesel_Manual,Mahindra,1.69,1
2655,CAR_002656,Toyota Innova 2.5 VX (Diesel) 8 Seater,2014,1200000.0,200000.0,Diesel,Individual,Manual,Second Owner,2656,11,Premium,Very High,0.0,Diesel_Manual,Toyota,6.0,1
2656,CAR_002657,Ford Fiesta Petrol Trend,2012,450000.0,60000.0,Petrol,Dealer,Manual,First Owner,2657,13,Mid,Medium,0.0,Petrol_Manual,Ford,7.5,0
2657,CAR_002658,Tata Tiago 1.2 Revotron XE,2019,425000.0,3000.0,Petrol,Individual,Manual,First Owner,2658,6,Low,Low,0.0,Petrol_Manual,Tata,141.67,0
2658,CAR_002659,Maruti Alto LXi,2006,150000.0,75118.0,Petrol,Individual,Manual,Second Owner,2659,19,Low,High,0.0,Petrol_Manual,Maruti,2.0,0
2659,CAR_002660,Maruti Ritz LDi,2013,4461000.0,140000.0,Diesel,Individual,Manual,Second Owner,2660,12,Low,High,0.0,Diesel_Manual,Maruti,2.07,1
2660,CAR_002661,Hyundai Xcent 1.2 Kappa Base,2014,335000.0,60000.0,Electric,Individual,Manual,Second Owner,2661,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,5.58,0
2661,CAR_002662,Chevrolet Optra 1.6,2006,120000.0,79000.0,Petrol,Dealer,Manual,Fourth & Above Owner,2662,19,Low,High,0.0,Petrol_Manual,Chevrolet,1.52,0
2662,CAR_002663,Ford Ikon 1.6 ZXI NXt,2005,4461000.0,25000.0,Petrol,Individual,Manual,Second Owner,2663,20,Low,Low,0.0,Petrol_Manual,Ford,0.8,0
2663,CAR_002664,Hyundai Getz GLS,2005,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,2664,20,Low,Medium,0.0,Petrol_Manual,Hyundai,3.57,0
2664,CAR_002665,Hyundai Santro Xing GL Plus,2008,4461000.0,41723.0,Petrol,Individual,Manual,Second Owner,2665,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.88,0
2665,CAR_002666,Maruti SX4 Vxi BSIV,2010,250000.0,70000.0,Petrol,Individual,Manual,First Owner,2666,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.57,0
2666,CAR_002667,Tata Indigo LS,2012,220000.0,70000.0,Diesel,Individual,Manual,Second Owner,2667,13,Low,High,0.0,Diesel_Manual,Tata,3.14,1
2667,CAR_002668,Ford Ikon 1.3 Flair,2005,61000.0,4637.0,Petrol,Individual,Manual,Second Owner,2668,20,Low,Low,0.0,Petrol_Manual,Ford,13.16,0
2668,CAR_002669,Maruti Swift VDI BSIV,2016,620000.0,70000.0,Diesel,Individual,Manual,First Owner,2669,9,High,Medium,0.0,Diesel_Manual,Maruti,8.86,1
2669,CAR_002670,Maruti Swift Dzire LDIX Limited Edition,2014,425000.0,100000.0,Diesel,Individual,Manual,Second Owner,2670,11,Low,High,0.0,Diesel_Manual,Maruti,4.25,1
2670,CAR_002671,Tata Indica Vista Aqua 1.3 Quadrajet,2009,85000.0,120000.0,Diesel,Individual,Manual,Second Owner,2671,16,Low,High,0.0,Diesel_Manual,Tata,0.71,1
2671,CAR_002672,Maruti Swift 1.3 DLX,2015,270000.0,40000.0,Diesel,Individual,Manual,Second Owner,2672,10,Low,High,0.0,Diesel_Manual,Maruti,6.75,1
2672,CAR_002673,Maruti Swift Vdi BSIII,2009,180000.0,220000.0,Diesel,Individual,Manual,First Owner,2673,16,Low,Very High,0.0,Diesel_Manual,Maruti,0.82,1
2673,CAR_002674,Renault KWID 1.0 RXT Optional,2018,300000.0,60000.0,Petrol,Individual,Manual,First Owner,2674,7,Low,Low,0.0,Petrol_Manual,Renault,15.0,0
2674,CAR_002675,Ford EcoSport 1.5 Petrol Titanium BSIV,2017,800000.0,19000.0,Petrol,Individual,Manual,First Owner,2675,8,High,Low,0.0,Petrol_Manual,Ford,42.11,0
2675,CAR_002676,Fiat Grande Punto Active (Diesel),2009,125000.0,95000.0,Diesel,Individual,Manual,Second Owner,2676,16,Low,High,0.0,Diesel_Manual,Fiat,1.32,1
2676,CAR_002677,Mahindra KUV 100 mFALCON G80 K2,2017,450000.0,60000.0,Petrol,Individual,Manual,First Owner,2677,8,Mid,Low,0.0,Petrol_Manual,Mahindra,15.0,0
2677,CAR_002678,Tata New Safari DICOR 2.2 VX 4x4,2012,450000.0,42655.0,Diesel,Individual,Manual,First Owner,2678,13,Mid,High,0.0,Diesel_Manual,Tata,10.55,1
2678,CAR_002679,Renault KWID 1.0 RXL,2017,250000.0,43000.0,Petrol,Individual,Manual,Second Owner,2679,8,Low,Medium,0.0,Petrol_Manual,Renault,5.81,0
2679,CAR_002680,Toyota Innova 2.5 VX 8 STR BSIV,2012,725000.0,70000.0,Diesel,Individual,Manual,First Owner,2680,13,High,Medium,0.0,Diesel_Manual,Toyota,10.36,1
2680,CAR_002681,Hyundai i20 Magna,2010,250000.0,55000.0,CNG,Individual,Manual,First Owner,2681,15,Low,Medium,0.0,Petrol_Manual,Hyundai,4.55,0
2681,CAR_002682,Fiat Punto 1.3 Active,2010,99000.0,80000.0,Diesel,Individual,Manual,Second Owner,2682,15,Low,High,0.0,Diesel_Manual,Fiat,1.24,1
2682,CAR_002683,Renault KWID 1.0,2017,350000.0,55000.0,Petrol,Individual,Manual,First Owner,2683,8,Mid,High,0.0,Petrol_Manual,Renault,6.36,0
2683,CAR_002684,Hyundai Xcent 1.2 VTVT E Plus,2018,511000.0,70000.0,Petrol,Individual,Manual,First Owner,2684,7,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.3,0
2684,CAR_002685,Maruti Ritz VDi,2012,300000.0,80000.0,Diesel,Individual,Manual,First Owner,2685,13,Low,High,0.0,Diesel_Manual,Maruti,3.75,1
2685,CAR_002686,Audi Q5 2.0 TFSI Quattro,2010,1100000.0,110000.0,Petrol,Individual,Manual,First Owner,2686,15,Premium,High,0.0,Petrol_Automatic,Audi,10.0,0
2686,CAR_002687,Hyundai i10 Sportz 1.2,2010,125000.0,60000.0,Petrol,Individual,Manual,Third Owner,2687,15,Low,Medium,0.0,Petrol_Manual,Hyundai,2.08,0
2687,CAR_002688,Tata Tiago 1.2 Revotron XZ,2018,500000.0,30000.0,Petrol,Individual,Manual,First Owner,2688,7,Mid,Low,0.0,Petrol_Manual,Tata,16.67,0
2688,CAR_002689,Renault Duster 85PS Diesel RxL,2013,385000.0,80000.0,Diesel,Individual,Manual,Second Owner,2689,12,Mid,High,0.0,Diesel_Manual,Renault,4.81,1
2689,CAR_002690,Skoda Rapid 1.6 MPI Elegance,2012,4461000.0,90000.0,Petrol,Individual,Manual,First Owner,2690,13,Mid,High,0.0,Petrol_Manual,Skoda,3.89,0
2690,CAR_002691,Maruti 800 Std BSIII,2003,60000.0,50000.0,Petrol,Individual,Manual,First Owner,2691,22,Low,Medium,0.0,Petrol_Manual,Maruti,1.2,0
2691,CAR_002692,Hyundai Santro Xing GLS,2012,300000.0,50000.0,Petrol,Individual,Manual,First Owner,2692,13,Low,Medium,0.0,Petrol_Manual,Hyundai,6.0,0
2692,CAR_002693,Hyundai i20 1.4 CRDi Asta,2012,400000.0,70000.0,Diesel,Individual,Manual,First Owner,2693,13,Mid,Medium,0.0,Diesel_Manual,Hyundai,5.71,1
2693,CAR_002694,Ford Ecosport 1.5 DV5 MT Titanium,2014,550000.0,110000.0,Diesel,Individual,Manual,First Owner,2694,11,Mid,High,0.0,Diesel_Manual,Ford,5.0,1
2694,CAR_002695,Nissan Sunny XL,2012,300000.0,80000.0,Petrol,Individual,Manual,Second Owner,2695,13,Low,High,0.0,Petrol_Manual,Nissan,3.75,0
2695,CAR_002696,Maruti Swift VDI,2012,250000.0,60000.0,Diesel,Individual,Manual,First Owner,2696,13,Low,Very High,0.0,Diesel_Manual,Maruti,1.6,1
2696,CAR_002697,Toyota Innova 2.5 GX 7 STR,2012,800000.0,180000.0,Diesel,Individual,Manual,First Owner,2697,13,High,Very High,0.0,Diesel_Manual,Toyota,4.44,1
2697,CAR_002698,Maruti Eeco 5 Seater AC BSIV,2018,300000.0,25000.0,Petrol,Individual,Manual,First Owner,2698,7,Low,Low,0.0,Petrol_Manual,Maruti,12.0,0
2698,CAR_002699,Maruti Alto LX,2004,80000.0,120000.0,Petrol,Individual,Manual,First Owner,2699,21,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
2699,CAR_002700,Mahindra Scorpio S5 BSIV,2020,1200000.0,11000.0,Diesel,Individual,Manual,First Owner,2700,5,Premium,Low,0.0,Diesel_Manual,Mahindra,109.09,1
2700,CAR_002701,Renault KWID AMT,2017,355000.0,40000.0,Petrol,Individual,Automatic,First Owner,2701,8,Mid,Medium,0.0,Petrol_Automatic,Renault,8.88,0
2701,CAR_002702,Maruti Swift Dzire VDI,2015,520000.0,50000.0,Diesel,Individual,Manual,First Owner,2702,10,Mid,Medium,0.0,Diesel_Manual,Maruti,10.4,1
2702,CAR_002703,Hyundai Verna 1.6 CRDI,2012,500000.0,26000.0,Diesel,Individual,Manual,First Owner,2703,13,Mid,Low,0.0,Diesel_Manual,Hyundai,19.23,1
2703,CAR_002704,Maruti Alto LXi,2008,114999.0,30000.0,Petrol,Individual,Manual,First Owner,2704,17,Low,Low,0.0,Petrol_Manual,Maruti,3.83,0
2704,CAR_002705,Toyota Fortuner 4x2 Manual,2014,4461000.0,50000.0,Diesel,Individual,Manual,First Owner,2705,11,Premium,Medium,0.0,Diesel_Manual,Toyota,36.0,1
2705,CAR_002706,Mahindra TUV 300 T6 Plus,2018,800000.0,60000.0,Diesel,Individual,Manual,First Owner,2706,7,High,Low,0.0,Diesel_Manual,Mahindra,40.0,1
2706,CAR_002707,Maruti Alto LXi BSIII,2008,105000.0,30000.0,Petrol,Individual,Manual,Second Owner,2707,17,Low,Low,0.0,Petrol_Manual,Maruti,3.5,0
2707,CAR_002708,Maruti Alto LXi,2011,135000.0,78000.0,Petrol,Individual,Manual,First Owner,2708,14,Low,High,0.0,Petrol_Manual,Maruti,1.73,0
2708,CAR_002709,Maruti Wagon R VXI BS IV,2011,170000.0,40000.0,Petrol,Individual,Manual,Second Owner,2709,14,Low,Medium,0.0,Petrol_Manual,Maruti,4.25,0
2709,CAR_002710,Hyundai Verna 1.6 SX VTVT,2014,550000.0,100000.0,Petrol,Individual,Manual,First Owner,2710,11,Mid,High,0.0,Petrol_Manual,Hyundai,5.5,0
2710,CAR_002711,Honda City 1.5 V MT,2009,250000.0,70000.0,Petrol,Individual,Manual,Third Owner,2711,16,Low,Medium,0.0,Petrol_Manual,Honda,3.57,0
2711,CAR_002712,Hyundai Elantra CRDi SX,2014,650000.0,30000.0,Diesel,Individual,Manual,Second Owner,2712,11,High,Low,0.0,Diesel_Manual,Hyundai,21.67,1
2712,CAR_002713,Honda City 1.5 V AT Exclusive,2008,210000.0,77000.0,Petrol,Individual,Automatic,Third Owner,2713,17,Low,High,0.0,Petrol_Automatic,Honda,2.73,0
2713,CAR_002714,Volkswagen Vento Diesel Highline,2011,260000.0,150000.0,Diesel,Individual,Manual,First Owner,2714,14,Low,High,0.0,Diesel_Manual,Volkswagen,1.73,1
2714,CAR_002715,Maruti Swift Dzire ZDI,2016,750000.0,60000.0,LPG,Individual,Manual,First Owner,2715,9,High,Medium,0.0,Diesel_Manual,Maruti,10.87,1
2715,CAR_002716,Toyota Innova Crysta 2.8 GX AT BSIV,2016,1350000.0,60000.0,Diesel,Individual,Automatic,First Owner,2716,9,Premium,High,0.0,Diesel_Automatic,Toyota,11.25,1
2716,CAR_002717,Maruti Ritz VDi,2012,220000.0,120000.0,Diesel,Individual,Manual,Second Owner,2717,13,Low,High,0.0,Diesel_Manual,Maruti,1.83,1
2717,CAR_002718,Hyundai Getz 1.3 GLS,2008,110000.0,110000.0,Electric,Individual,Manual,Second Owner,2718,17,Low,High,0.0,Petrol_Manual,Hyundai,1.0,0
2718,CAR_002719,Mahindra XUV500 AT W8 FWD,2017,1230000.0,60000.0,Petrol,Individual,Automatic,First Owner,2719,8,Premium,Medium,0.0,Diesel_Automatic,Mahindra,17.57,1
2719,CAR_002720,Mahindra Quanto C6,2013,220000.0,60000.0,Diesel,Individual,Manual,First Owner,2720,12,Low,High,0.0,Diesel_Manual,Mahindra,2.0,1
2720,CAR_002721,Tata Indigo LX Dicor,2009,130000.0,50000.0,Diesel,Individual,Manual,First Owner,2721,16,Low,Medium,0.0,Diesel_Manual,Tata,2.6,1
2721,CAR_002722,Hyundai i20 1.2 Magna,2012,330000.0,44000.0,Petrol,Individual,Manual,First Owner,2722,13,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.5,0
2722,CAR_002723,Honda City 1.5 S MT,2010,254999.0,90000.0,Petrol,Individual,Manual,Second Owner,2723,15,Low,High,0.0,Petrol_Manual,Honda,2.83,0
2723,CAR_002724,Hyundai Verna 1.6 SX VTVT (O),2014,600000.0,50000.0,Diesel,Individual,Manual,Second Owner,2724,11,Low,High,0.0,Petrol_Manual,Hyundai,12.0,0
2724,CAR_002725,Hyundai i10 Magna,2012,210000.0,50000.0,Petrol,Individual,Manual,First Owner,2725,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.2,0
2725,CAR_002726,Chevrolet Enjoy TCDi LT 7 Seater,2014,300000.0,146000.0,Diesel,Individual,Manual,First Owner,2726,11,Low,High,0.0,Diesel_Manual,Chevrolet,2.05,1
2726,CAR_002727,Tata Tiago 1.2 Revotron XTA,2018,426000.0,15000.0,Petrol,Individual,Automatic,First Owner,2727,7,Mid,Low,0.0,Petrol_Automatic,Tata,28.4,0
2727,CAR_002728,Maruti Alto K10 VXI,2015,265000.0,60000.0,Petrol,Individual,Manual,First Owner,2728,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.42,0
2728,CAR_002729,Hyundai Xcent 1.2 Kappa SX,2016,450000.0,35000.0,CNG,Individual,Manual,Second Owner,2729,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,12.86,0
2729,CAR_002730,Hyundai Verna 1.6 SX,2012,480000.0,80000.0,Diesel,Individual,Manual,First Owner,2730,13,Mid,High,0.0,Diesel_Manual,Hyundai,6.0,1
2730,CAR_002731,Tata Nano Lx,2010,40000.0,90000.0,Petrol,Individual,Manual,First Owner,2731,15,Low,High,0.0,Petrol_Manual,Tata,0.44,0
2731,CAR_002732,Honda City 1.5 S MT,2011,395000.0,70000.0,Petrol,Individual,Manual,Second Owner,2732,14,Mid,Medium,0.0,Petrol_Manual,Honda,5.64,0
2732,CAR_002733,Maruti Wagon R VXI BS IV,2013,229999.0,62000.0,Petrol,Individual,Manual,Second Owner,2733,12,Low,Medium,0.0,Petrol_Manual,Maruti,3.71,0
2733,CAR_002734,Ford Ecosport 1.5 Petrol Titanium Plus AT,2018,875000.0,10000.0,Petrol,Individual,Automatic,First Owner,2734,7,High,Low,0.0,Petrol_Automatic,Ford,87.5,0
2734,CAR_002735,Maruti Eeco 5 STR With AC Plus HTR CNG,2010,170000.0,80000.0,CNG,Individual,Manual,Fourth & Above Owner,2735,15,Low,High,0.0,CNG_Manual,Maruti,2.12,0
2735,CAR_002736,Hyundai i20 Sportz 1.2,2015,500000.0,35000.0,Petrol,Individual,Manual,Second Owner,2736,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,14.29,0
2736,CAR_002737,Renault Duster 85PS Diesel RxL,2013,450000.0,1000.0,Diesel,Dealer,Manual,Second Owner,2737,12,Mid,High,0.0,Diesel_Manual,Renault,450.0,1
2737,CAR_002738,Mercedes-Benz C-Class Progressive C 220d,2018,3800000.0,10000.0,Diesel,Dealer,Automatic,First Owner,2738,7,Premium,Low,0.0,Diesel_Automatic,Mercedes-Benz,380.0,1
2738,CAR_002739,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,2739,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
2739,CAR_002740,BMW X5 xDrive 30d xLine,2019,4950000.0,60000.0,Diesel,Dealer,Automatic,First Owner,2740,6,Premium,Low,0.0,Diesel_Automatic,BMW,165.0,1
2740,CAR_002741,Hyundai Creta 1.6 CRDi SX,2016,535000.0,60000.0,Diesel,Individual,Manual,First Owner,2741,9,Mid,High,0.0,Diesel_Manual,Hyundai,10.17,1
2741,CAR_002742,Maruti SX4 Vxi BSIV,2012,225000.0,110000.0,Petrol,Individual,Manual,Second Owner,2742,13,Low,High,0.0,Petrol_Manual,Maruti,2.05,0
2742,CAR_002743,Hyundai Grand i10 1.2 Kappa Magna AT,2017,4461000.0,19890.0,Petrol,Dealer,Automatic,First Owner,2743,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
2743,CAR_002744,Maruti Swift ZXI BSIV,2016,670000.0,7104.0,Petrol,Trustmark Dealer,Manual,First Owner,2744,9,High,Low,0.0,Petrol_Manual,Maruti,94.31,0
2744,CAR_002745,Maruti Ertiga VXI,2015,625000.0,11918.0,Petrol,Trustmark Dealer,Manual,First Owner,2745,10,High,Low,0.0,Petrol_Manual,Maruti,52.44,0
2745,CAR_002746,Hyundai Grand i10 Magna AT,2017,520000.0,10510.0,Petrol,Dealer,Automatic,First Owner,2746,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,49.48,0
2746,CAR_002747,Chevrolet Beat LT Option,2016,239000.0,41000.0,Petrol,Dealer,Manual,First Owner,2747,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.83,0
2747,CAR_002748,Toyota Fortuner 4x2 AT,2017,2600000.0,47162.0,Diesel,Trustmark Dealer,Manual,First Owner,2748,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,55.13,1
2748,CAR_002749,Maruti S-Cross Zeta DDiS 200 SH,2015,750000.0,45974.0,Diesel,Trustmark Dealer,Manual,First Owner,2749,10,High,Medium,0.0,Diesel_Manual,Maruti,16.31,1
2749,CAR_002750,Hyundai i10 Magna,2012,229999.0,49824.0,Petrol,Dealer,Manual,First Owner,2750,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.62,0
2750,CAR_002751,Audi A6 2.0 TDI Premium Plus,2013,1300000.0,58500.0,Diesel,Dealer,Automatic,First Owner,2751,12,Premium,Medium,0.0,Diesel_Automatic,Audi,22.22,1
2751,CAR_002752,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,2752,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.41,0
2752,CAR_002753,Skoda Laura Ambiente 2.0 TDI CR MT,2012,385000.0,52000.0,Diesel,Dealer,Manual,First Owner,2753,13,Mid,Medium,0.0,Diesel_Manual,Skoda,7.4,1
2753,CAR_002754,Hyundai Verna 1.6 VTVT SX,2015,4461000.0,55340.0,Petrol,Trustmark Dealer,Manual,First Owner,2754,10,Low,Medium,0.0,Petrol_Manual,Hyundai,13.73,0
2754,CAR_002755,Maruti Swift Dzire VDI,2017,4461000.0,46507.0,LPG,Trustmark Dealer,Manual,First Owner,2755,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.9,1
2755,CAR_002756,Maruti Ignis 1.2 Sigma BSIV,2019,450000.0,40000.0,Petrol,Individual,Manual,First Owner,2756,6,Mid,Medium,0.0,Petrol_Manual,Maruti,11.25,0
2756,CAR_002757,Maruti Ignis 1.2 Sigma BSIV,2019,450000.0,40000.0,Petrol,Individual,Manual,First Owner,2757,6,Mid,Medium,0.0,Petrol_Manual,Maruti,11.25,0
2757,CAR_002758,Maruti Ertiga VDI,2012,550000.0,120000.0,Diesel,Individual,Manual,Third Owner,2758,13,Mid,High,0.0,Diesel_Manual,Maruti,4.58,1
2758,CAR_002759,Tata Manza Aura Safire BS IV,2013,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,2759,12,Low,Medium,0.0,Petrol_Manual,Tata,3.33,0
2759,CAR_002760,Tata Indica GLS BS IV,2008,80000.0,90000.0,Petrol,Individual,Manual,Second Owner,2760,17,Low,High,0.0,Petrol_Manual,Tata,0.89,0
2760,CAR_002761,Tata New Safari DICOR 2.2 EX 4x2,2010,300000.0,250000.0,Diesel,Individual,Manual,Second Owner,2761,15,Low,Very High,0.0,Diesel_Manual,Tata,1.2,1
2761,CAR_002762,Ford Aspire Titanium Plus Diesel BSIV,2019,675000.0,15000.0,Electric,Individual,Manual,First Owner,2762,6,High,Low,0.0,Diesel_Manual,Ford,45.0,1
2762,CAR_002763,Maruti Eeco 7 Seater Standard BSIV,2017,260000.0,30000.0,Petrol,Individual,Manual,First Owner,2763,8,Low,Low,0.0,Petrol_Manual,Maruti,8.67,0
2763,CAR_002764,Hyundai Grand i10 1.2 Kappa Asta,2019,500000.0,15000.0,Petrol,Individual,Manual,First Owner,2764,6,Mid,Low,0.0,Petrol_Manual,Hyundai,33.33,0
2764,CAR_002765,Hyundai EON Sportz,2012,150000.0,55766.0,Petrol,Individual,Manual,Second Owner,2765,13,Low,Medium,0.0,Petrol_Manual,Hyundai,2.69,0
2765,CAR_002766,Maruti Alto LXi,2006,100000.0,80000.0,Petrol,Individual,Manual,First Owner,2766,19,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
2766,CAR_002767,Maruti Wagon R LXI,2001,70000.0,117000.0,Petrol,Individual,Manual,Second Owner,2767,24,Low,High,0.0,Petrol_Manual,Maruti,0.6,0
2767,CAR_002768,Maruti Alto LX,2004,100000.0,60000.0,Petrol,Individual,Manual,First Owner,2768,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.43,0
2768,CAR_002769,Toyota Innova 2.5 VX (Diesel) 7 Seater,2012,730000.0,110000.0,Diesel,Individual,Manual,Second Owner,2769,13,High,High,0.0,Diesel_Manual,Toyota,6.64,1
2769,CAR_002770,Nissan Terrano XL 85 PS,2016,525000.0,100000.0,Diesel,Individual,Manual,Second Owner,2770,9,Mid,High,0.0,Diesel_Manual,Nissan,5.25,1
2770,CAR_002771,Nissan Terrano XL 85 PS,2016,525000.0,105000.0,Diesel,Individual,Manual,Second Owner,2771,9,Mid,High,0.0,Diesel_Manual,Nissan,5.0,1
2771,CAR_002772,Toyota Innova 2.5 V Diesel 7-seater,2009,550000.0,182000.0,Diesel,Individual,Manual,Third Owner,2772,16,Mid,Very High,0.0,Diesel_Manual,Toyota,3.02,1
2772,CAR_002773,Toyota Corolla Altis Diesel D4DJ,2012,420000.0,60000.0,Petrol,Individual,Manual,Second Owner,2773,13,Mid,High,0.0,Diesel_Manual,Toyota,4.42,1
2773,CAR_002774,Honda Amaze SX i-VTEC,2015,500000.0,70000.0,Petrol,Individual,Manual,First Owner,2774,10,Mid,Medium,0.0,Petrol_Manual,Honda,7.14,0
2774,CAR_002775,Maruti Wagon R LXI Minor,2009,165000.0,76000.0,Petrol,Individual,Manual,Second Owner,2775,16,Low,High,0.0,Petrol_Manual,Maruti,2.17,0
2775,CAR_002776,Hyundai Verna 1.6 SX,2014,650000.0,70000.0,Diesel,Individual,Manual,Second Owner,2776,11,High,Medium,0.0,Diesel_Manual,Hyundai,9.29,1
2776,CAR_002777,Chevrolet Beat Diesel LT,2013,195000.0,46000.0,Diesel,Individual,Manual,First Owner,2777,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,4.24,1
2777,CAR_002778,Maruti Zen LX,1999,75000.0,70000.0,Petrol,Individual,Manual,First Owner,2778,26,Low,Medium,0.0,Petrol_Manual,Maruti,1.07,0
2778,CAR_002779,Hyundai Accent CRDi,2005,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,2779,20,Low,High,0.0,Diesel_Manual,Hyundai,0.83,1
2779,CAR_002780,Toyota Innova Crysta 2.4 VX MT BSIV,2017,1770000.0,25000.0,Diesel,Individual,Manual,First Owner,2780,8,Premium,Low,0.0,Diesel_Manual,Toyota,70.8,1
2780,CAR_002781,Maruti Esteem Lxi - BSIII,2006,4461000.0,90000.0,Petrol,Individual,Manual,First Owner,2781,19,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
2781,CAR_002782,Hyundai Xcent 1.1 CRDi Base,2016,515000.0,50000.0,Diesel,Individual,Manual,First Owner,2782,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.3,1
2782,CAR_002783,Honda Jazz 1.5 VX i DTEC,2017,700000.0,24585.0,Diesel,Dealer,Manual,Test Drive Car,2783,8,High,Low,0.0,Diesel_Manual,Honda,28.47,1
2783,CAR_002784,Maruti Zen LX,2001,62000.0,70000.0,Petrol,Individual,Manual,First Owner,2784,24,Low,Medium,0.0,Petrol_Manual,Maruti,0.89,0
2784,CAR_002785,Toyota Etios GD SP,2011,220000.0,110000.0,Diesel,Individual,Manual,Third Owner,2785,14,Low,High,0.0,Diesel_Manual,Toyota,2.0,1
2785,CAR_002786,Hyundai Santro Magna CNG BSIV,2019,520000.0,10000.0,CNG,Individual,Manual,First Owner,2786,6,Mid,Low,0.0,CNG_Manual,Hyundai,52.0,0
2786,CAR_002787,Fiat Linea Emotion,2010,170000.0,100000.0,Petrol,Individual,Manual,First Owner,2787,15,Low,High,0.0,Petrol_Manual,Fiat,1.7,0
2787,CAR_002788,Ford Figo Diesel Titanium,2011,190000.0,110000.0,Diesel,Individual,Manual,Second Owner,2788,14,Low,High,0.0,Diesel_Manual,Ford,1.73,1
2788,CAR_002789,Ford Figo Diesel Titanium,2011,150000.0,100000.0,Diesel,Individual,Manual,Second Owner,2789,14,Low,High,0.0,Diesel_Manual,Ford,1.5,1
2789,CAR_002790,Maruti Zen Estilo LXI BS IV,2009,125000.0,65000.0,Petrol,Individual,Manual,First Owner,2790,16,Low,Medium,0.0,Petrol_Manual,Maruti,1.92,0
2790,CAR_002791,Honda Amaze EX i-Dtech,2015,290000.0,40000.0,Diesel,Individual,Manual,Third Owner,2791,10,Low,Medium,0.0,Diesel_Manual,Honda,7.25,1
2791,CAR_002792,Maruti Celerio X ZXI BSIV,2019,490000.0,13900.0,Petrol,Individual,Manual,First Owner,2792,6,Mid,Low,0.0,Petrol_Manual,Maruti,35.25,0
2792,CAR_002793,Tata Tigor 1.2 Revotron XZ Option,2017,434999.0,17563.0,Petrol,Individual,Manual,First Owner,2793,8,Mid,Low,0.0,Petrol_Manual,Tata,24.77,0
2793,CAR_002794,Tata Zest Revotron 1.2T XE,2016,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,2794,9,Mid,High,0.0,Petrol_Manual,Tata,25.33,0
2794,CAR_002795,Volkswagen Polo Diesel Trendline 1.2L,2012,4461000.0,60000.0,Diesel,Individual,Manual,Second Owner,2795,13,Low,High,0.0,Diesel_Manual,Volkswagen,2.44,1
2795,CAR_002796,Ford Figo 1.5D Trend MT,2016,495000.0,40000.0,Diesel,Individual,Manual,First Owner,2796,9,Mid,Medium,0.0,Diesel_Manual,Ford,12.38,1
2796,CAR_002797,Toyota Innova 2.5 Z Diesel 7 Seater BS IV,2014,894999.0,173000.0,Diesel,Individual,Manual,Third Owner,2797,11,High,Very High,0.0,Diesel_Manual,Toyota,5.17,1
2797,CAR_002798,Mahindra XUV500 W6 2WD,2014,600000.0,151000.0,CNG,Individual,Manual,Second Owner,2798,11,Mid,High,0.0,Diesel_Manual,Mahindra,3.97,1
2798,CAR_002799,Honda Brio S MT,2015,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,2799,10,Low,Medium,0.0,Petrol_Manual,Honda,6.0,0
2799,CAR_002800,Honda Amaze E i-VTEC,2017,430000.0,15000.0,Petrol,Individual,Manual,First Owner,2800,8,Low,High,0.0,Petrol_Manual,Honda,28.67,0
2800,CAR_002801,Honda Amaze E i-DTEC,2017,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,2801,8,Mid,High,0.0,Diesel_Manual,Honda,4.62,1
2801,CAR_002802,Maruti Baleno Zeta 1.2,2019,4461000.0,11000.0,LPG,Individual,Manual,First Owner,2802,6,High,Low,0.0,Petrol_Manual,Maruti,68.18,0
2802,CAR_002803,Maruti Alto K10 LXI CNG Optional,2016,300000.0,50000.0,CNG,Individual,Manual,First Owner,2803,9,Low,Medium,0.0,CNG_Manual,Maruti,6.0,0
2803,CAR_002804,BMW X1 sDrive20d,2012,700000.0,120000.0,Diesel,Individual,Automatic,Second Owner,2804,13,High,High,0.0,Diesel_Automatic,BMW,5.83,1
2804,CAR_002805,Maruti Wagon R LXI DUO BSIII,2007,100000.0,70000.0,LPG,Individual,Manual,Third Owner,2805,18,Low,Medium,0.0,LPG_Manual,Maruti,1.43,0
2805,CAR_002806,Hyundai Grand i10 Sportz,2015,300000.0,90000.0,Petrol,Individual,Manual,First Owner,2806,10,Low,High,0.0,Petrol_Manual,Hyundai,3.33,0
2806,CAR_002807,Tata Indica Vista Quadrajet VX,2012,130000.0,185000.0,Diesel,Individual,Manual,Second Owner,2807,13,Low,Very High,0.0,Diesel_Manual,Tata,0.7,1
2807,CAR_002808,Hyundai i20 Asta Option 1.4 CRDi,2016,650000.0,40000.0,Diesel,Individual,Manual,First Owner,2808,9,High,Medium,0.0,Diesel_Manual,Hyundai,16.25,1
2808,CAR_002809,Maruti Swift VDI BSIV,2017,509999.0,25000.0,Diesel,Individual,Manual,First Owner,2809,8,Mid,Low,0.0,Diesel_Manual,Maruti,20.4,1
2809,CAR_002810,Toyota Innova 2.5 VX (Diesel) 7 Seater,2013,1000000.0,117780.0,Diesel,Individual,Manual,Second Owner,2810,12,High,High,0.0,Diesel_Manual,Toyota,8.49,1
2810,CAR_002811,Hyundai Verna CRDi SX,2010,280000.0,120000.0,Diesel,Individual,Manual,Second Owner,2811,15,Low,High,0.0,Diesel_Manual,Hyundai,2.33,1
2811,CAR_002812,Maruti Swift DDiS VDI,2017,600000.0,120000.0,Diesel,Individual,Manual,First Owner,2812,8,Mid,High,0.0,Diesel_Manual,Maruti,5.0,1
2812,CAR_002813,Skoda Rapid 1.6 TDI Ambition,2013,4461000.0,60000.0,Diesel,Individual,Manual,Third Owner,2813,12,Low,High,0.0,Diesel_Manual,Skoda,1.75,1
2813,CAR_002814,Audi Q3 2.0 TDI Quattro Premium Plus,2014,1150000.0,110000.0,Diesel,Individual,Automatic,Second Owner,2814,11,Premium,High,0.0,Diesel_Automatic,Audi,10.45,1
2814,CAR_002815,Hyundai EON Magna Plus,2018,229999.0,30000.0,Petrol,Individual,Manual,First Owner,2815,7,Low,Low,0.0,Petrol_Manual,Hyundai,7.67,0
2815,CAR_002816,Hyundai EON Magna Plus,2018,245000.0,30000.0,Petrol,Individual,Manual,First Owner,2816,7,Low,Low,0.0,Petrol_Manual,Hyundai,8.17,0
2816,CAR_002817,Mahindra Scorpio S4 4WD,2015,600000.0,60000.0,Diesel,Individual,Manual,First Owner,2817,10,Low,Medium,0.0,Diesel_Manual,Mahindra,10.0,1
2817,CAR_002818,Hyundai Grand i10 Sportz,2016,375000.0,25000.0,Petrol,Individual,Manual,First Owner,2818,9,Mid,Low,0.0,Petrol_Manual,Hyundai,15.0,0
2818,CAR_002819,Mahindra Scorpio 1.99 S10,2015,150000.0,55000.0,Diesel,Individual,Manual,Second Owner,2819,10,Low,High,0.0,Diesel_Manual,Mahindra,2.73,1
2819,CAR_002820,Mahindra XUV500 W10 2WD,2018,1450000.0,20000.0,Electric,Individual,Manual,First Owner,2820,7,Premium,Low,0.0,Diesel_Manual,Mahindra,72.5,1
2820,CAR_002821,Maruti Ertiga VDI,2014,500000.0,100000.0,Petrol,Individual,Manual,Second Owner,2821,11,Mid,High,0.0,Diesel_Manual,Maruti,5.0,1
2821,CAR_002822,Maruti Alto LXi,2007,4461000.0,40000.0,Petrol,Individual,Manual,Second Owner,2822,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.88,0
2822,CAR_002823,Hyundai Sonata AT Leather,2006,225000.0,60000.0,Petrol,Individual,Manual,Third Owner,2823,19,Low,High,0.0,Petrol_Automatic,Hyundai,2.81,0
2823,CAR_002824,Hyundai Sonata 2.4L AT,2006,275000.0,80000.0,Petrol,Individual,Automatic,Fourth & Above Owner,2824,19,Low,High,0.0,Petrol_Automatic,Hyundai,3.44,0
2824,CAR_002825,Mahindra Bolero Power Plus SLX,2018,620000.0,35000.0,Diesel,Individual,Manual,First Owner,2825,7,Low,Medium,0.0,Diesel_Manual,Mahindra,17.71,1
2825,CAR_002826,Nissan X-Trail SLX MT,2010,600000.0,110000.0,Diesel,Individual,Manual,Second Owner,2826,15,Mid,High,0.0,Diesel_Manual,Nissan,5.45,1
2826,CAR_002827,Mahindra Xylo E8 ABS Airbag BSIV,2011,275000.0,160000.0,Diesel,Individual,Manual,First Owner,2827,14,Low,Very High,0.0,Diesel_Manual,Mahindra,1.72,1
2827,CAR_002828,Tata Indigo TDI,2013,125000.0,125000.0,Diesel,Individual,Manual,First Owner,2828,12,Low,High,0.0,Diesel_Manual,Tata,1.0,1
2828,CAR_002829,Tata Indigo GLS,2011,80000.0,62000.0,Petrol,Individual,Manual,Second Owner,2829,14,Low,Medium,0.0,Petrol_Manual,Tata,1.29,0
2829,CAR_002830,Nissan Micra Diesel XV,2011,199000.0,120000.0,Diesel,Individual,Manual,Second Owner,2830,14,Low,High,0.0,Diesel_Manual,Nissan,1.66,1
2830,CAR_002831,Nissan Micra XL,2016,440000.0,33000.0,Petrol,Individual,Manual,Second Owner,2831,9,Low,Medium,0.0,Petrol_Manual,Nissan,13.33,0
2831,CAR_002832,Maruti Swift Dzire LDI,2013,250000.0,80000.0,Diesel,Individual,Manual,Fourth & Above Owner,2832,12,Low,High,0.0,Diesel_Manual,Maruti,3.12,1
2832,CAR_002833,Hyundai i20 1.4 Sportz,2017,700000.0,35000.0,Diesel,Individual,Manual,First Owner,2833,8,Low,Medium,0.0,Diesel_Manual,Hyundai,20.0,1
2833,CAR_002834,Hyundai Grand i10 CRDi Asta Option,2015,450000.0,50000.0,Diesel,Individual,Manual,First Owner,2834,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,9.0,1
2834,CAR_002835,Maruti Alto 800 Std Optional,2019,240000.0,60000.0,Petrol,Individual,Manual,First Owner,2835,6,Low,High,0.0,Petrol_Manual,Maruti,48.0,0
2835,CAR_002836,Hyundai Xcent 1.2 Kappa SX,2016,350000.0,12000.0,Petrol,Individual,Manual,First Owner,2836,9,Mid,Low,0.0,Petrol_Manual,Hyundai,29.17,0
2836,CAR_002837,Ford EcoSport 1.5 TDCi Titanium BSIV,2016,650000.0,81595.0,Diesel,Dealer,Manual,First Owner,2837,9,High,High,0.0,Diesel_Manual,Ford,7.97,1
2837,CAR_002838,Hyundai Getz GLS ABS,2005,71000.0,22000.0,CNG,Individual,Manual,First Owner,2838,20,Low,Low,0.0,Petrol_Manual,Hyundai,3.23,0
2838,CAR_002839,Maruti Swift VXI Optional,2015,400000.0,60000.0,Petrol,Individual,Manual,First Owner,2839,10,Mid,Medium,0.0,Petrol_Manual,Maruti,10.0,0
2839,CAR_002840,Maruti Alto LXi BSIII,2008,130000.0,40000.0,LPG,Individual,Manual,Second Owner,2840,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.25,0
2840,CAR_002841,Mahindra Scorpio 2.6 CRDe SLE,2013,4461000.0,70000.0,Diesel,Individual,Manual,Second Owner,2841,12,Low,Medium,0.0,Diesel_Manual,Mahindra,4.29,1
2841,CAR_002842,Nissan Kicks XL D BSIV,2019,1100000.0,4000.0,Diesel,Individual,Manual,First Owner,2842,6,Premium,Low,0.0,Diesel_Manual,Nissan,275.0,1
2842,CAR_002843,BMW 3 Series 320d Luxury Line,2015,2200000.0,24000.0,Diesel,Individual,Automatic,First Owner,2843,10,Premium,Low,0.0,Diesel_Automatic,BMW,91.67,1
2843,CAR_002844,Honda Amaze VX i-VTEC,2018,800000.0,13500.0,Electric,Dealer,Manual,First Owner,2844,7,High,Low,0.0,Petrol_Manual,Honda,59.26,0
2844,CAR_002845,Maruti Ritz VDI (ABS) BS IV,2013,490000.0,40000.0,Diesel,Individual,Manual,First Owner,2845,12,Mid,Medium,0.0,Diesel_Manual,Maruti,12.25,1
2845,CAR_002846,Honda Jazz 1.2 VX i VTEC,2018,836000.0,9700.0,Petrol,Dealer,Manual,First Owner,2846,7,High,Low,0.0,Petrol_Manual,Honda,86.19,0
2846,CAR_002847,Volkswagen Polo Petrol Comfortline 1.2L,2012,375000.0,62000.0,Petrol,Individual,Manual,Second Owner,2847,13,Mid,Medium,0.0,Petrol_Manual,Volkswagen,6.05,0
2847,CAR_002848,Honda BR-V i-DTEC VX MT,2016,1249000.0,24000.0,Diesel,Dealer,Manual,First Owner,2848,9,Premium,Low,0.0,Diesel_Manual,Honda,52.04,1
2848,CAR_002849,Honda WR-V i-DTEC VX,2019,1240000.0,13000.0,Petrol,Dealer,Manual,First Owner,2849,6,Premium,Low,0.0,Diesel_Manual,Honda,95.38,1
2849,CAR_002850,Tata Nano Std BSII,2009,35000.0,50000.0,Petrol,Individual,Manual,Third Owner,2850,16,Low,Medium,0.0,Petrol_Manual,Tata,0.7,0
2850,CAR_002851,Honda BR-V i-VTEC VX MT,2017,1068000.0,36000.0,Petrol,Dealer,Manual,First Owner,2851,8,Premium,Medium,0.0,Petrol_Manual,Honda,29.67,0
2851,CAR_002852,Honda BR-V i-DTEC VX MT,2017,1189000.0,40000.0,Diesel,Dealer,Manual,First Owner,2852,8,Premium,High,0.0,Diesel_Manual,Honda,29.72,1
2852,CAR_002853,Hyundai Elite i20 Petrol Asta Option,2018,700000.0,27000.0,Petrol,Individual,Manual,First Owner,2853,7,High,Low,0.0,Petrol_Manual,Hyundai,25.93,0
2853,CAR_002854,Maruti Alto K10 VXI,2016,350000.0,40000.0,Petrol,Individual,Manual,First Owner,2854,9,Mid,Medium,0.0,Petrol_Manual,Maruti,8.75,0
2854,CAR_002855,Hyundai i10 Magna 1.2,2009,254999.0,80000.0,CNG,Individual,Manual,Second Owner,2855,16,Low,High,0.0,Petrol_Manual,Hyundai,3.19,0
2855,CAR_002856,Mahindra Scorpio 2.6 CRDe,2005,229999.0,60000.0,Diesel,Individual,Manual,Third Owner,2856,20,Low,Very High,0.0,Diesel_Manual,Mahindra,1.04,1
2856,CAR_002857,Hyundai i20 Asta Option 1.4 CRDi,2016,750000.0,80000.0,Diesel,Individual,Manual,First Owner,2857,9,High,High,0.0,Diesel_Manual,Hyundai,9.38,1
2857,CAR_002858,Maruti Ritz VDi,2012,325000.0,119000.0,LPG,Individual,Manual,Second Owner,2858,13,Mid,High,0.0,Diesel_Manual,Maruti,2.73,1
2858,CAR_002859,Toyota Innova 2.5 G4 Diesel 7-seater,2009,760000.0,190000.0,Diesel,Individual,Manual,Second Owner,2859,16,High,Very High,0.0,Diesel_Manual,Toyota,4.0,1
2859,CAR_002860,Maruti Celerio ZXI AT,2017,509999.0,25000.0,Petrol,Individual,Automatic,First Owner,2860,8,Mid,Low,0.0,Petrol_Automatic,Maruti,20.4,0
2860,CAR_002861,Nissan Micra XL,2013,4461000.0,28740.0,Petrol,Individual,Manual,First Owner,2861,12,Mid,Low,0.0,Petrol_Manual,Nissan,12.63,0
2861,CAR_002862,Maruti Alto K10 LXI,2010,229999.0,60000.0,Petrol,Individual,Manual,First Owner,2862,15,Low,Medium,0.0,Petrol_Manual,Maruti,4.74,0
2862,CAR_002863,Nissan Sunny XL,2012,290000.0,110000.0,Petrol,Individual,Manual,Second Owner,2863,13,Low,High,0.0,Petrol_Manual,Nissan,2.64,0
2863,CAR_002864,Maruti Swift VXI,2018,550000.0,30000.0,Petrol,Individual,Manual,First Owner,2864,7,Mid,Low,0.0,Petrol_Manual,Maruti,18.33,0
2864,CAR_002865,Tata Indica Vista TDI LX,2011,125000.0,120000.0,Diesel,Individual,Manual,Second Owner,2865,14,Low,High,0.0,Diesel_Manual,Tata,1.04,1
2865,CAR_002866,Volvo XC60 D5 Inscription,2014,2000000.0,130000.0,Electric,Individual,Automatic,First Owner,2866,11,Premium,High,0.0,Diesel_Automatic,Volvo,15.38,1
2866,CAR_002867,Toyota Innova 2.5 G4 Diesel 7-seater,2008,350000.0,140000.0,Diesel,Individual,Manual,Second Owner,2867,17,Low,High,0.0,Diesel_Manual,Toyota,2.5,1
2867,CAR_002868,Hyundai Verna 1.6 SX CRDi (O),2011,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,2868,14,Low,High,0.0,Diesel_Manual,Hyundai,4.33,1
2868,CAR_002869,Maruti Wagon R VXI Minor ABS,2010,260000.0,50000.0,Petrol,Individual,Manual,First Owner,2869,15,Low,Medium,0.0,Petrol_Manual,Maruti,5.2,0
2869,CAR_002870,Maruti Omni BSIII 8-STR W/ IMMOBILISER,2008,150000.0,50000.0,Petrol,Individual,Manual,Second Owner,2870,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
2870,CAR_002871,Maruti Esteem Vxi - BSIII,2006,85000.0,60000.0,Petrol,Individual,Manual,Second Owner,2871,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.42,0
2871,CAR_002872,Maruti Alto 800 LXI,2015,280000.0,15000.0,Petrol,Individual,Manual,Second Owner,2872,10,Low,Low,0.0,Petrol_Manual,Maruti,18.67,0
2872,CAR_002873,Maruti Celerio VXI,2015,220000.0,45000.0,Petrol,Individual,Manual,First Owner,2873,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.89,0
2873,CAR_002874,Hyundai Santro Xing XG,2005,140000.0,80000.0,Petrol,Individual,Manual,Third Owner,2874,20,Low,High,0.0,Petrol_Manual,Hyundai,1.75,0
2874,CAR_002875,Maruti Wagon R AMT VXI Option,2018,370000.0,15000.0,Petrol,Individual,Automatic,First Owner,2875,7,Mid,Low,0.0,Petrol_Automatic,Maruti,24.67,0
2875,CAR_002876,Maruti Swift Dzire VDI,2019,650000.0,20000.0,Diesel,Individual,Manual,First Owner,2876,6,High,Low,0.0,Diesel_Manual,Maruti,32.5,1
2876,CAR_002877,Maruti Alto 800 LX,2017,280000.0,37000.0,Diesel,Individual,Manual,First Owner,2877,8,Low,Medium,0.0,Petrol_Manual,Maruti,7.57,0
2877,CAR_002878,Maruti Swift Dzire VDI,2016,540000.0,80000.0,Diesel,Individual,Manual,First Owner,2878,9,Mid,High,0.0,Diesel_Manual,Maruti,6.75,1
2878,CAR_002879,Ambassador Classic 2000 Dsz,2002,50000.0,120000.0,Diesel,Individual,Manual,Fourth & Above Owner,2879,23,Low,High,0.0,Diesel_Manual,Ambassador,0.42,1
2879,CAR_002880,Maruti Ciaz Sigma BSIV,2018,800000.0,20000.0,Petrol,Individual,Manual,First Owner,2880,7,High,Low,0.0,Petrol_Manual,Maruti,40.0,0
2880,CAR_002881,Ford Figo 1.5D Titanium MT,2016,515000.0,25000.0,Diesel,Individual,Manual,First Owner,2881,9,Mid,Low,0.0,Diesel_Manual,Ford,20.6,1
2881,CAR_002882,Hyundai i20 1.2 Magna,2010,220000.0,96000.0,CNG,Individual,Manual,Third Owner,2882,15,Low,High,0.0,Petrol_Manual,Hyundai,2.29,0
2882,CAR_002883,Hyundai i20 Asta (o),2014,525000.0,54000.0,Petrol,Dealer,Manual,Second Owner,2883,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.72,0
2883,CAR_002884,Chevrolet Spark 1.0 LT Option Pack w/ Airbag,2013,165000.0,60000.0,Petrol,Dealer,Manual,First Owner,2884,12,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.11,0
2884,CAR_002885,Fiat Linea 1.3 Emotion,2010,221000.0,65000.0,LPG,Dealer,Manual,First Owner,2885,15,Low,Medium,0.0,Diesel_Manual,Fiat,3.4,1
2885,CAR_002886,Chevrolet Beat LT,2015,245000.0,43000.0,Petrol,Dealer,Manual,First Owner,2886,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,5.7,0
2886,CAR_002887,Ford Figo 1.2P Titanium Plus MT,2012,4461000.0,42000.0,Petrol,Dealer,Manual,First Owner,2887,13,Low,Medium,0.0,Petrol_Manual,Ford,6.19,0
2887,CAR_002888,Skoda Superb Elegance 2.0 TDI CR AT,2011,550000.0,73000.0,Diesel,Dealer,Manual,First Owner,2888,14,Mid,High,0.0,Diesel_Automatic,Skoda,7.53,1
2888,CAR_002889,Hyundai Grand i10 1.2 Kappa Era,2015,434999.0,29000.0,Petrol,Dealer,Manual,First Owner,2889,10,Mid,Low,0.0,Petrol_Manual,Hyundai,15.0,0
2889,CAR_002890,Tata Safari Storme EX,2013,450000.0,95000.0,Diesel,Dealer,Manual,First Owner,2890,12,Mid,High,0.0,Diesel_Manual,Tata,4.74,1
2890,CAR_002891,Chevrolet Sail 1.2 Base,2014,290000.0,25000.0,Petrol,Individual,Manual,First Owner,2891,11,Low,High,0.0,Petrol_Manual,Chevrolet,11.6,0
2891,CAR_002892,Volkswagen Vento 1.0 TSI Highline Plus,2011,305000.0,53000.0,Electric,Dealer,Manual,Second Owner,2892,14,Mid,Medium,0.0,Petrol_Manual,Volkswagen,5.75,0
2892,CAR_002893,Ford Classic 1.6 Duratec LXI,2011,265000.0,20000.0,Petrol,Dealer,Manual,First Owner,2893,14,Low,Low,0.0,Petrol_Manual,Ford,13.25,0
2893,CAR_002894,Honda Brio 1.2 S MT,2014,4461000.0,57000.0,Petrol,Dealer,Manual,First Owner,2894,11,Mid,Medium,0.0,Petrol_Manual,Honda,5.7,0
2894,CAR_002895,Chevrolet Cruze LTZ,2014,550000.0,52000.0,Diesel,Dealer,Manual,First Owner,2895,11,Mid,Medium,0.0,Diesel_Manual,Chevrolet,10.58,1
2895,CAR_002896,Maruti Zen Estilo LXI BSIII,2008,105000.0,40000.0,Diesel,Individual,Manual,First Owner,2896,17,Low,Medium,0.0,Petrol_Manual,Maruti,2.62,0
2896,CAR_002897,Honda Brio 1.2 S MT,2012,275000.0,47000.0,Petrol,Dealer,Manual,First Owner,2897,13,Low,Medium,0.0,Petrol_Manual,Honda,5.85,0
2897,CAR_002898,Honda City 1.5 V MT,2012,434999.0,60000.0,Petrol,Dealer,Manual,Second Owner,2898,13,Mid,Medium,0.0,Petrol_Manual,Honda,9.26,0
2898,CAR_002899,Hyundai Verna 1.6 CRDi SX,2016,690000.0,73000.0,CNG,Dealer,Manual,First Owner,2899,9,High,High,0.0,Diesel_Manual,Hyundai,9.45,1
2899,CAR_002900,Honda Brio V MT,2012,249000.0,42000.0,Petrol,Dealer,Manual,Second Owner,2900,13,Low,Medium,0.0,Petrol_Manual,Honda,5.93,0
2900,CAR_002901,Hyundai EON D Lite Plus,2012,204999.0,60000.0,Petrol,Dealer,Manual,Third Owner,2901,13,Low,High,0.0,Petrol_Manual,Hyundai,3.42,0
2901,CAR_002902,Chevrolet Beat Diesel LT,2012,200000.0,50000.0,Diesel,Individual,Manual,First Owner,2902,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,4.0,1
2902,CAR_002903,Maruti Wagon R VXI Optional,2015,190000.0,120000.0,Petrol,Individual,Manual,First Owner,2903,10,Low,High,0.0,Petrol_Manual,Maruti,1.58,0
2903,CAR_002904,Maruti Alto LXi,2012,150000.0,50000.0,Petrol,Individual,Manual,First Owner,2904,13,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
2904,CAR_002905,Maruti Zen LX,1999,70000.0,70000.0,Petrol,Individual,Manual,Fourth & Above Owner,2905,26,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
2905,CAR_002906,Mahindra Bolero Power Plus SLX,2018,900000.0,25000.0,Diesel,Individual,Manual,First Owner,2906,7,High,Low,0.0,Diesel_Manual,Mahindra,36.0,1
2906,CAR_002907,Maruti Ertiga ZXI AT Petrol,2019,950000.0,11000.0,Petrol,Individual,Automatic,First Owner,2907,6,High,Low,0.0,Petrol_Automatic,Maruti,86.36,0
2907,CAR_002908,Maruti Swift Dzire VXI,2013,475000.0,80000.0,Petrol,Individual,Manual,First Owner,2908,12,Low,High,0.0,Petrol_Manual,Maruti,5.94,0
2908,CAR_002909,Maruti 800 Std,1998,55000.0,80000.0,Petrol,Individual,Manual,Fourth & Above Owner,2909,27,Low,High,0.0,Petrol_Manual,Maruti,0.69,0
2909,CAR_002910,Ford Fiesta Titanium 1.5 TDCi,2013,450000.0,75000.0,Diesel,Individual,Manual,Second Owner,2910,12,Mid,High,0.0,Diesel_Manual,Ford,6.0,1
2910,CAR_002911,BMW 5 Series 525d Sedan,2009,1200000.0,45000.0,Diesel,Individual,Automatic,Third Owner,2911,16,Premium,Medium,0.0,Diesel_Automatic,BMW,26.67,1
2911,CAR_002912,Mahindra Bolero SLX,2007,300000.0,100000.0,LPG,Individual,Manual,Second Owner,2912,18,Low,High,0.0,Diesel_Manual,Mahindra,3.0,1
2912,CAR_002913,Honda City i DTEC E,2014,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,2913,11,Mid,High,0.0,Diesel_Manual,Honda,5.83,1
2913,CAR_002914,Tata Indica Vista Quadrajet LS,2011,229999.0,160000.0,Diesel,Individual,Manual,Second Owner,2914,14,Low,Very High,0.0,Diesel_Manual,Tata,1.44,1
2914,CAR_002915,Hyundai Creta 1.6 Gamma SX Plus,2015,821000.0,50000.0,Petrol,Individual,Manual,First Owner,2915,10,High,Medium,0.0,Petrol_Manual,Hyundai,16.42,0
2915,CAR_002916,Maruti Alto K10 VXI,2014,200000.0,80000.0,Petrol,Individual,Manual,Second Owner,2916,11,Low,High,0.0,Petrol_Manual,Maruti,2.5,0
2916,CAR_002917,Mahindra Scorpio 1.99 S10,2017,1000000.0,60000.0,Diesel,Individual,Manual,Second Owner,2917,8,High,Medium,0.0,Diesel_Manual,Mahindra,16.67,1
2917,CAR_002918,Maruti Alto 800 VXI,2016,4461000.0,60000.0,Petrol,Individual,Manual,Second Owner,2918,9,Low,Medium,0.0,Petrol_Manual,Maruti,3.67,0
2918,CAR_002919,Ford Figo Aspire 1.5 TDCi Trend,2015,459999.0,60000.0,Diesel,Dealer,Manual,First Owner,2919,10,Mid,High,0.0,Diesel_Manual,Ford,3.1,1
2919,CAR_002920,Maruti SX4 ZDI,2012,310000.0,70000.0,Diesel,Individual,Manual,Second Owner,2920,13,Mid,Medium,0.0,Diesel_Manual,Maruti,4.43,1
2920,CAR_002921,Hyundai Elite i20 Diesel Asta Option,2018,815000.0,42000.0,Diesel,Individual,Manual,First Owner,2921,7,High,Medium,0.0,Diesel_Manual,Hyundai,19.4,1
2921,CAR_002922,Hyundai Verna 1.6 CRDI SX Option,2017,1200000.0,40000.0,Diesel,Individual,Manual,First Owner,2922,8,Premium,Medium,0.0,Diesel_Manual,Hyundai,30.0,1
2922,CAR_002923,Chevrolet Optra Magnum 2.0 LT,2011,4461000.0,150000.0,Diesel,Individual,Manual,Second Owner,2923,14,Low,High,0.0,Diesel_Manual,Chevrolet,1.13,1
2923,CAR_002924,Maruti S-Cross Alpha DDiS 200 SH,2017,1000000.0,30000.0,Diesel,Individual,Manual,First Owner,2924,8,High,Low,0.0,Diesel_Manual,Maruti,33.33,1
2924,CAR_002925,Hyundai Santro Xing GLS,2010,155000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,2925,15,Low,High,0.0,Petrol_Manual,Hyundai,1.72,0
2925,CAR_002926,Mahindra TUV 300 T6 Plus,2017,400000.0,120000.0,Diesel,Individual,Manual,First Owner,2926,8,Mid,High,0.0,Diesel_Manual,Mahindra,3.33,1
2926,CAR_002927,Mahindra Quanto C4,2016,400000.0,80000.0,Diesel,Individual,Manual,Second Owner,2927,9,Mid,High,0.0,Diesel_Manual,Mahindra,5.0,1
2927,CAR_002928,Tata Indica Vista Aura 1.3 Quadrajet BSIV,2010,250000.0,80000.0,Diesel,Individual,Manual,Second Owner,2928,15,Low,High,0.0,Diesel_Manual,Tata,3.12,1
2928,CAR_002929,Tata Indigo GLX,2008,150000.0,50000.0,Petrol,Individual,Manual,Third Owner,2929,17,Low,High,0.0,Petrol_Manual,Tata,3.0,0
2929,CAR_002930,Hyundai Santro LP zipPlus,2002,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,2930,23,Low,Medium,0.0,Petrol_Manual,Hyundai,1.14,0
2930,CAR_002931,Maruti SX4 ZXI AT,2009,130000.0,60000.0,Petrol,Individual,Automatic,Second Owner,2931,16,Low,High,0.0,Petrol_Automatic,Maruti,1.08,0
2931,CAR_002932,Volkswagen Vento 1.5 TDI Comfortline AT,2014,325000.0,70000.0,Diesel,Individual,Automatic,First Owner,2932,11,Mid,Medium,0.0,Diesel_Automatic,Volkswagen,4.64,1
2932,CAR_002933,Volkswagen Jetta 2.0L TDI Highline AT,2012,735000.0,55300.0,Diesel,Dealer,Automatic,First Owner,2933,13,High,Medium,0.0,Diesel_Automatic,Volkswagen,13.29,1
2933,CAR_002934,Tata Indica Vista Quadrajet 90 VX,2012,285000.0,74300.0,Diesel,Dealer,Manual,First Owner,2934,13,Low,High,0.0,Diesel_Manual,Tata,3.84,1
2934,CAR_002935,Honda City VX MT,2010,350000.0,48781.0,Petrol,Dealer,Manual,First Owner,2935,15,Mid,Medium,0.0,Petrol_Manual,Honda,7.17,0
2935,CAR_002936,Volkswagen Jetta 1.9 Highline TDI,2010,290000.0,87620.0,Diesel,Dealer,Automatic,First Owner,2936,15,Low,High,0.0,Diesel_Automatic,Volkswagen,3.31,1
2936,CAR_002937,Volkswagen Vento 1.5 TDI Highline Plus AT,2017,890000.0,40219.0,Diesel,Dealer,Automatic,First Owner,2937,8,High,Medium,0.0,Diesel_Automatic,Volkswagen,22.13,1
2937,CAR_002938,Honda Jazz VX,2011,385000.0,11473.0,Petrol,Dealer,Manual,First Owner,2938,14,Low,High,0.0,Petrol_Manual,Honda,33.56,0
2938,CAR_002939,Maruti Eeco 5 Seater AC BSIV,2017,425000.0,8352.0,Petrol,Dealer,Manual,First Owner,2939,8,Mid,Low,0.0,Petrol_Manual,Maruti,50.89,0
2939,CAR_002940,Maruti Wagon R VXI AMT1.2BSIV,2017,525000.0,9745.0,Petrol,Dealer,Automatic,First Owner,2940,8,Mid,Low,0.0,Petrol_Automatic,Maruti,53.87,0
2940,CAR_002941,Hyundai Grand i10 Asta,2017,550000.0,9748.0,Petrol,Dealer,Manual,First Owner,2941,8,Mid,Low,0.0,Petrol_Manual,Hyundai,56.42,0
2941,CAR_002942,Volkswagen Polo 1.0 MPI Trendline,2012,271000.0,49000.0,Petrol,Dealer,Manual,First Owner,2942,13,Low,Medium,0.0,Petrol_Manual,Volkswagen,5.53,0
2942,CAR_002943,Hyundai Creta 1.6 CRDi SX Option,2018,1490000.0,20694.0,Electric,Dealer,Manual,First Owner,2943,7,Premium,Low,0.0,Diesel_Manual,Hyundai,72.0,1
2943,CAR_002944,Hyundai Grand i10 Asta Option,2015,4461000.0,31080.0,Petrol,Dealer,Manual,First Owner,2944,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,15.77,0
2944,CAR_002945,Hyundai EON Era Plus,2014,260000.0,37605.0,Petrol,Dealer,Manual,First Owner,2945,11,Low,Medium,0.0,Petrol_Manual,Hyundai,6.91,0
2945,CAR_002946,Hyundai Xcent 1.1 CRDi SX Option,2014,455000.0,60000.0,Diesel,Dealer,Manual,First Owner,2946,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,8.15,1
2946,CAR_002947,Tata Nano Lx BSIV,2010,75000.0,58850.0,Petrol,Dealer,Manual,First Owner,2947,15,Low,Medium,0.0,Petrol_Manual,Tata,1.27,0
2947,CAR_002948,Toyota Etios Cross 1.2L G,2015,421000.0,23839.0,Petrol,Dealer,Manual,First Owner,2948,10,Mid,Low,0.0,Petrol_Manual,Toyota,17.66,0
2948,CAR_002949,Maruti Swift DDiS LDI,2016,550000.0,54000.0,Diesel,Dealer,Manual,First Owner,2949,9,Mid,Medium,0.0,Diesel_Manual,Maruti,10.19,1
2949,CAR_002950,Hyundai i10 Sportz 1.2 AT,2013,330000.0,38000.0,Petrol,Dealer,Automatic,First Owner,2950,12,Mid,Medium,0.0,Petrol_Automatic,Hyundai,8.68,0
2950,CAR_002951,Volkswagen Vento 1.5 TDI Comfortline,2012,390000.0,45454.0,Diesel,Dealer,Manual,First Owner,2951,13,Mid,Medium,0.0,Diesel_Manual,Volkswagen,8.58,1
2951,CAR_002952,Skoda Rapid 1.5 TDI AT Ambition,2015,599000.0,46957.0,Diesel,Dealer,Manual,First Owner,2952,10,Mid,Medium,0.0,Diesel_Automatic,Skoda,12.76,1
2952,CAR_002953,Hyundai Elite i20 Sportz Plus CVT BSIV,2019,738000.0,8000.0,Petrol,Individual,Manual,First Owner,2953,6,High,Low,0.0,Petrol_Automatic,Hyundai,92.25,0
2953,CAR_002954,Maruti Alto K10 VXI,2016,300000.0,30000.0,Petrol,Individual,Manual,First Owner,2954,9,Low,Low,0.0,Petrol_Manual,Maruti,10.0,0
2954,CAR_002955,Tata Indica Vista Quadrajet LX,2012,150000.0,110000.0,Diesel,Individual,Manual,First Owner,2955,13,Low,High,0.0,Diesel_Manual,Tata,1.36,1
2955,CAR_002956,Toyota Innova 2.5 G4 Diesel 7-seater,2007,440000.0,223000.0,Diesel,Individual,Manual,Fourth & Above Owner,2956,18,Mid,Very High,0.0,Diesel_Manual,Toyota,1.97,1
2956,CAR_002957,Volkswagen Vento IPL II Diesel Trendline,2011,280000.0,110000.0,Diesel,Individual,Manual,First Owner,2957,14,Low,High,0.0,Diesel_Manual,Volkswagen,2.55,1
2957,CAR_002958,Hyundai Santro GS,2006,100000.0,120000.0,Petrol,Individual,Manual,Second Owner,2958,19,Low,High,0.0,Petrol_Manual,Hyundai,0.83,0
2958,CAR_002959,Maruti Swift VXi BSIV,2008,150000.0,100000.0,CNG,Individual,Manual,Second Owner,2959,17,Low,High,0.0,Petrol_Manual,Maruti,1.5,0
2959,CAR_002960,Maruti Swift Dzire VDI,2014,380000.0,100000.0,Diesel,Individual,Manual,Second Owner,2960,11,Mid,High,0.0,Diesel_Manual,Maruti,3.8,1
2960,CAR_002961,Maruti Alto LX,2006,90000.0,60000.0,Petrol,Individual,Manual,Second Owner,2961,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.5,0
2961,CAR_002962,Mahindra Scorpio VLS 2.2 mHawk,2008,300000.0,270000.0,Diesel,Individual,Manual,Third Owner,2962,17,Low,Very High,0.0,Diesel_Manual,Mahindra,1.11,1
2962,CAR_002963,Skoda Fabia 1.2 TDI Active,2011,275000.0,60000.0,Diesel,Individual,Manual,Second Owner,2963,14,Low,Medium,0.0,Diesel_Manual,Skoda,4.58,1
2963,CAR_002964,Maruti Zen Estilo Sports,2009,180000.0,41090.0,Petrol,Individual,Manual,First Owner,2964,16,Low,Medium,0.0,Petrol_Manual,Maruti,4.38,0
2964,CAR_002965,Maruti Swift VDI,2012,225000.0,296823.0,Diesel,Individual,Manual,First Owner,2965,13,Low,Very High,0.0,Diesel_Manual,Maruti,0.76,1
2965,CAR_002966,Maruti Alto 800 LXI,2014,200000.0,40000.0,Petrol,Individual,Manual,First Owner,2966,11,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
2966,CAR_002967,Toyota Corolla Altis G,2011,500000.0,40000.0,Petrol,Individual,Manual,Second Owner,2967,14,Mid,Medium,0.0,Petrol_Manual,Toyota,12.5,0
2967,CAR_002968,Ford Fiesta 1.6 Duratec S,2010,300000.0,70000.0,Petrol,Individual,Manual,Second Owner,2968,15,Low,Medium,0.0,Petrol_Manual,Ford,4.29,0
2968,CAR_002969,Hyundai Verna 1.6 CRDi SX,2015,765000.0,80000.0,Diesel,Individual,Manual,Second Owner,2969,10,High,High,0.0,Diesel_Manual,Hyundai,9.56,1
2969,CAR_002970,Maruti Swift Dzire VDI,2017,4461000.0,90000.0,Diesel,Individual,Manual,First Owner,2970,8,High,High,0.0,Diesel_Manual,Maruti,7.22,1
2970,CAR_002971,Maruti Wagon R VXI Plus Optional,2017,380000.0,19000.0,Petrol,Individual,Manual,First Owner,2971,8,Mid,Low,0.0,Petrol_Manual,Maruti,20.0,0
2971,CAR_002972,Maruti Swift Dzire VDI,2014,480000.0,110000.0,Diesel,Individual,Manual,Second Owner,2972,11,Mid,High,0.0,Diesel_Manual,Maruti,4.36,1
2972,CAR_002973,Mahindra Jeep MM 540,1996,200000.0,60000.0,Diesel,Individual,Manual,First Owner,2973,29,Low,Medium,0.0,Diesel_Manual,Mahindra,3.33,1
2973,CAR_002974,Maruti Wagon R VXI Plus Optional,2017,380000.0,19000.0,LPG,Individual,Manual,First Owner,2974,8,Mid,Low,0.0,Petrol_Manual,Maruti,20.0,0
2974,CAR_002975,Maruti Zen Estilo LXI Green (CNG),2010,145000.0,68000.0,CNG,Individual,Manual,Second Owner,2975,15,Low,Medium,0.0,CNG_Manual,Maruti,2.13,0
2975,CAR_002976,Maruti Ritz VDi,2011,280000.0,120000.0,Diesel,Individual,Manual,Second Owner,2976,14,Low,High,0.0,Diesel_Manual,Maruti,2.33,1
2976,CAR_002977,Maruti Swift Dzire VDI,2014,254999.0,60000.0,Diesel,Individual,Manual,First Owner,2977,11,Low,High,0.0,Diesel_Manual,Maruti,8.5,1
2977,CAR_002978,Tata Bolt Revotron XE,2016,300000.0,90000.0,Petrol,Individual,Manual,Second Owner,2978,9,Low,High,0.0,Petrol_Manual,Tata,3.33,0
2978,CAR_002979,Maruti Swift VDI BSIV,2015,480000.0,140000.0,Diesel,Individual,Manual,Third Owner,2979,10,Mid,High,0.0,Diesel_Manual,Maruti,3.43,1
2979,CAR_002980,Skoda Rapid Monte Carlo 1.6 MPI AT BSIV,2017,850000.0,44500.0,Petrol,Individual,Automatic,First Owner,2980,8,High,Medium,0.0,Petrol_Automatic,Skoda,19.1,0
2980,CAR_002981,Hyundai i20 Asta 1.4 CRDi,2014,350000.0,52000.0,Diesel,Individual,Manual,Second Owner,2981,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,6.73,1
2981,CAR_002982,Hyundai i20 Active 1.2 SX,2016,650000.0,26000.0,Petrol,Individual,Manual,First Owner,2982,9,High,Low,0.0,Petrol_Manual,Hyundai,25.0,0
2982,CAR_002983,Hyundai i20 Active 1.2 SX,2016,650000.0,26000.0,Petrol,Individual,Manual,First Owner,2983,9,High,Low,0.0,Petrol_Manual,Hyundai,25.0,0
2983,CAR_002984,Tata New Safari 4X2,2007,550000.0,80000.0,Petrol,Individual,Manual,Second Owner,2984,18,Mid,High,0.0,Petrol_Manual,Tata,6.88,0
2984,CAR_002985,Maruti Celerio VXI,2015,300000.0,35000.0,Petrol,Individual,Manual,First Owner,2985,10,Low,Medium,0.0,Petrol_Manual,Maruti,8.57,0
2985,CAR_002986,Maruti Swift 1.3 VXi,2008,150000.0,60000.0,Electric,Individual,Manual,Second Owner,2986,17,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
2986,CAR_002987,Hyundai Grand i10 Sportz,2014,4461000.0,120000.0,Petrol,Individual,Manual,First Owner,2987,11,Low,High,0.0,Petrol_Manual,Hyundai,2.5,0
2987,CAR_002988,Hyundai Santro Xing XG,2004,80000.0,58000.0,Petrol,Individual,Manual,First Owner,2988,21,Low,Medium,0.0,Petrol_Manual,Hyundai,1.38,0
2988,CAR_002989,Tata Manza Club Class Quadrajet90 LS,2014,200000.0,100000.0,Diesel,Individual,Manual,First Owner,2989,11,Low,High,0.0,Diesel_Manual,Tata,2.0,1
2989,CAR_002990,Hyundai EON Magna Plus,2016,220000.0,43700.0,Petrol,Individual,Manual,First Owner,2990,9,Low,Medium,0.0,Petrol_Manual,Hyundai,5.03,0
2990,CAR_002991,Tata Indica GLS BS IV,2009,68000.0,120000.0,Petrol,Individual,Manual,Second Owner,2991,16,Low,High,0.0,Petrol_Manual,Tata,0.57,0
2991,CAR_002992,Tata Manza ELAN Quadrajet BS IV,2011,180000.0,80000.0,Diesel,Individual,Manual,Third Owner,2992,14,Low,High,0.0,Diesel_Manual,Tata,2.25,1
2992,CAR_002993,Hyundai EON Era Plus,2017,290000.0,60000.0,Petrol,Individual,Manual,First Owner,2993,8,Low,Low,0.0,Petrol_Manual,Hyundai,14.5,0
2993,CAR_002994,Maruti Eeco 5 Seater AC BSIV,2018,300000.0,25000.0,Petrol,Individual,Manual,First Owner,2994,7,Low,Low,0.0,Petrol_Manual,Maruti,12.0,0
2994,CAR_002995,Honda City i DTec SV,2014,4461000.0,27483.0,Diesel,Dealer,Manual,First Owner,2995,11,Mid,Low,0.0,Diesel_Manual,Honda,21.83,1
2995,CAR_002996,Maruti Swift LXI,2019,470000.0,8000.0,Petrol,Individual,Manual,First Owner,2996,6,Mid,Low,0.0,Petrol_Manual,Maruti,58.75,0
2996,CAR_002997,Hyundai i10 Sportz 1.2,2008,150000.0,30000.0,Petrol,Individual,Manual,First Owner,2997,17,Low,High,0.0,Petrol_Manual,Hyundai,5.0,0
2997,CAR_002998,Maruti Swift VVT VXI,2017,400000.0,30000.0,Petrol,Individual,Manual,First Owner,2998,8,Low,Low,0.0,Petrol_Manual,Maruti,13.33,0
2998,CAR_002999,Ford Fiesta 1.4 ZXi TDCi ABS,2009,110000.0,100000.0,Diesel,Individual,Manual,Fourth & Above Owner,2999,16,Low,High,0.0,Diesel_Manual,Ford,1.1,1
2999,CAR_003000,Maruti Alto 800 LXI,2014,180000.0,56207.0,Petrol,Dealer,Manual,Second Owner,3000,11,Low,Medium,0.0,Petrol_Manual,Maruti,3.2,0
3000,CAR_003001,Hyundai i20 Sportz Petrol,2010,250000.0,60000.0,Petrol,Individual,Manual,First Owner,3001,15,Low,Medium,0.0,Petrol_Manual,Hyundai,6.25,0
3001,CAR_003002,Maruti Zen LXI,2006,80000.0,70000.0,Petrol,Individual,Manual,First Owner,3002,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.14,0
3002,CAR_003003,Honda WR-V i-VTEC VX,2018,875000.0,1440.0,Petrol,Individual,Manual,First Owner,3003,7,High,Low,0.0,Petrol_Manual,Honda,607.64,0
3003,CAR_003004,Hyundai Grand i10 Sportz,2015,425000.0,12000.0,Petrol,Individual,Manual,First Owner,3004,10,Mid,Low,0.0,Petrol_Manual,Hyundai,35.42,0
3004,CAR_003005,Maruti Alto 800 LXI,2019,300000.0,50000.0,Petrol,Individual,Manual,First Owner,3005,6,Low,Medium,0.0,Petrol_Manual,Maruti,6.0,0
3005,CAR_003006,Honda Amaze VX i-DTEC,2016,700000.0,35000.0,Diesel,Individual,Manual,First Owner,3006,9,High,Medium,0.0,Diesel_Manual,Honda,20.0,1
3006,CAR_003007,Maruti Zen Estilo VXI BSIV,2010,130000.0,50000.0,Petrol,Individual,Manual,Second Owner,3007,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.6,0
3007,CAR_003008,Maruti Ertiga SHVS ZDI Plus,2017,1000000.0,60000.0,Diesel,Individual,Manual,Second Owner,3008,8,High,Medium,0.0,Diesel_Manual,Maruti,16.67,1
3008,CAR_003009,Maruti Swift Dzire ZDI,2017,650000.0,67000.0,Diesel,Individual,Manual,First Owner,3009,8,High,Medium,0.0,Diesel_Manual,Maruti,9.7,1
3009,CAR_003010,Renault Duster Petrol RxL,2013,400000.0,100000.0,Petrol,Individual,Manual,Second Owner,3010,12,Mid,High,0.0,Petrol_Manual,Renault,4.0,0
3010,CAR_003011,Mahindra TUV 300 mHAWK100 T8,2017,800000.0,60000.0,Diesel,Individual,Manual,First Owner,3011,8,High,Medium,0.0,Diesel_Manual,Mahindra,13.33,1
3011,CAR_003012,Mahindra Bolero Power Plus SLX,2017,675000.0,70000.0,Diesel,Individual,Manual,First Owner,3012,8,High,Medium,0.0,Diesel_Manual,Mahindra,9.64,1
3012,CAR_003013,Maruti Omni E MPI STD BS IV,2017,210000.0,50000.0,Petrol,Individual,Manual,Fourth & Above Owner,3013,8,Low,Medium,0.0,Petrol_Manual,Maruti,4.2,0
3013,CAR_003014,Tata New Safari DICOR 2.2 EX 4x2,2009,150000.0,120000.0,Diesel,Individual,Manual,Third Owner,3014,16,Low,High,0.0,Diesel_Manual,Tata,1.25,1
3014,CAR_003015,Maruti Omni CNG,2007,45000.0,100000.0,CNG,Individual,Manual,Fourth & Above Owner,3015,18,Low,High,0.0,CNG_Manual,Maruti,0.45,0
3015,CAR_003016,Hyundai i20 Sportz 1.2,2015,516000.0,90000.0,Petrol,Individual,Manual,First Owner,3016,10,Mid,High,0.0,Petrol_Manual,Hyundai,5.73,0
3016,CAR_003017,Hyundai i20 Sportz 1.2,2014,450000.0,80000.0,Petrol,Individual,Manual,First Owner,3017,11,Mid,High,0.0,Petrol_Manual,Hyundai,5.62,0
3017,CAR_003018,Chevrolet Spark 1.0 LS,2012,134000.0,60000.0,Petrol,Individual,Manual,Second Owner,3018,13,Low,High,0.0,Petrol_Manual,Chevrolet,1.49,0
3018,CAR_003019,Toyota Etios VD,2014,509999.0,157000.0,Diesel,Individual,Manual,Second Owner,3019,11,Mid,Very High,0.0,Diesel_Manual,Toyota,3.25,1
3019,CAR_003020,Volkswagen Polo 1.2 MPI Comfortline,2016,500000.0,20000.0,Petrol,Individual,Manual,First Owner,3020,9,Low,Low,0.0,Petrol_Manual,Volkswagen,25.0,0
3020,CAR_003021,Maruti Zen Estilo 1.1 LXI BSIII,2008,130000.0,80000.0,Petrol,Individual,Manual,Fourth & Above Owner,3021,17,Low,High,0.0,Petrol_Manual,Maruti,1.62,0
3021,CAR_003022,Maruti Swift ZDi BSIV,2015,575000.0,60000.0,Diesel,Individual,Manual,First Owner,3022,10,Mid,Medium,0.0,Diesel_Manual,Maruti,9.58,1
3022,CAR_003023,Hyundai Elite i20 Sportz Plus BSIV,2015,550000.0,49000.0,Petrol,Individual,Manual,First Owner,3023,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,11.22,0
3023,CAR_003024,Tata Indica Vista Terra TDI BSIII,2009,145000.0,89255.0,Diesel,Individual,Manual,First Owner,3024,16,Low,High,0.0,Diesel_Manual,Tata,1.62,1
3024,CAR_003025,Maruti Alto 800 LXI,2020,347000.0,60000.0,Petrol,Individual,Manual,First Owner,3025,5,Mid,Low,0.0,Petrol_Manual,Maruti,69.4,0
3025,CAR_003026,Tata Indica Vista Quadrajet VX,2013,275000.0,110000.0,Diesel,Individual,Manual,First Owner,3026,12,Low,High,0.0,Diesel_Manual,Tata,2.5,1
3026,CAR_003027,Toyota Innova 2.5 V Diesel 8-seater,2010,630000.0,60000.0,Diesel,Individual,Manual,Third Owner,3027,15,High,Very High,0.0,Diesel_Manual,Toyota,3.15,1
3027,CAR_003028,Chevrolet Beat Diesel,2012,160000.0,90000.0,Diesel,Individual,Manual,Second Owner,3028,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.78,1
3028,CAR_003029,Fiat Grande Punto 1.3 Dynamic (Diesel),2010,150000.0,63000.0,Diesel,Individual,Manual,Third Owner,3029,15,Low,Medium,0.0,Diesel_Manual,Fiat,2.38,1
3029,CAR_003030,Honda Mobilio S i VTEC,2015,4461000.0,23000.0,Petrol,Individual,Manual,First Owner,3030,10,Mid,Low,0.0,Petrol_Manual,Honda,23.91,0
3030,CAR_003031,Mahindra Quanto C6,2013,349000.0,60000.0,Diesel,Individual,Manual,First Owner,3031,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.49,1
3031,CAR_003032,Mahindra Bolero DI DX 8 Seater,2004,229999.0,168000.0,Diesel,Individual,Manual,Fourth & Above Owner,3032,21,Low,Very High,0.0,Diesel_Manual,Mahindra,1.37,1
3032,CAR_003033,Tata Indica Vista Aqua 1.4 TDI,2008,4461000.0,90000.0,Diesel,Individual,Manual,Third Owner,3033,17,Low,High,0.0,Diesel_Manual,Tata,1.22,1
3033,CAR_003034,Maruti Wagon R LXI Minor,2008,4461000.0,70000.0,Diesel,Individual,Manual,Third Owner,3034,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.57,0
3034,CAR_003035,Renault KWID RXE,2018,245000.0,5550.0,Petrol,Individual,Manual,Second Owner,3035,7,Low,Low,0.0,Petrol_Manual,Renault,44.14,0
3035,CAR_003036,Maruti Alto K10 VXI,2014,180000.0,100000.0,Petrol,Individual,Manual,Second Owner,3036,11,Low,High,0.0,Petrol_Manual,Maruti,1.8,0
3036,CAR_003037,Maruti Wagon R VXI BS IV,2014,290000.0,80000.0,Petrol,Individual,Manual,Second Owner,3037,11,Low,High,0.0,Petrol_Manual,Maruti,3.62,0
3037,CAR_003038,Hyundai i20 Magna,2010,250000.0,40000.0,Petrol,Individual,Manual,First Owner,3038,15,Low,Medium,0.0,Petrol_Manual,Hyundai,6.25,0
3038,CAR_003039,Tata Indigo TDI,2010,150000.0,100000.0,Diesel,Individual,Manual,First Owner,3039,15,Low,High,0.0,Diesel_Manual,Tata,1.5,1
3039,CAR_003040,Maruti 800 AC BSII,2004,55000.0,35000.0,Petrol,Individual,Manual,Third Owner,3040,21,Low,Medium,0.0,Petrol_Manual,Maruti,1.57,0
3040,CAR_003041,Skoda Laura L n K 1.9 PD,2006,500000.0,75000.0,Diesel,Individual,Manual,Second Owner,3041,19,Mid,High,0.0,Diesel_Manual,Skoda,6.67,1
3041,CAR_003042,Mahindra Quanto C4,2013,290000.0,150000.0,Diesel,Individual,Manual,Fourth & Above Owner,3042,12,Low,High,0.0,Diesel_Manual,Mahindra,1.93,1
3042,CAR_003043,Hyundai Santro Magna AMT BSIV,2019,400000.0,16000.0,Petrol,Individual,Automatic,First Owner,3043,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,25.0,0
3043,CAR_003044,Maruti Alto STD,2009,72000.0,60000.0,Petrol,Individual,Manual,First Owner,3044,16,Low,Medium,0.0,Petrol_Manual,Maruti,1.03,0
3044,CAR_003045,Maruti Ritz VDi,2014,250000.0,115000.0,Diesel,Individual,Manual,First Owner,3045,11,Low,High,0.0,Diesel_Manual,Maruti,2.17,1
3045,CAR_003046,Maruti Swift AMT VXI,2019,4461000.0,60000.0,Petrol,Individual,Automatic,First Owner,3046,6,Mid,Low,0.0,Petrol_Automatic,Maruti,23.0,0
3046,CAR_003047,Kia Seltos HTK Plus AT D,2019,1300000.0,10000.0,Diesel,Individual,Automatic,First Owner,3047,6,Premium,Low,0.0,Diesel_Automatic,Kia,130.0,1
3047,CAR_003048,Hyundai Santro Magna BSIV,2019,425000.0,15000.0,Petrol,Individual,Manual,First Owner,3048,6,Mid,Low,0.0,Petrol_Manual,Hyundai,28.33,0
3048,CAR_003049,Maruti Eeco CNG 5 Seater AC BSIV,2019,470000.0,10000.0,CNG,Individual,Manual,First Owner,3049,6,Mid,Low,0.0,CNG_Manual,Maruti,47.0,0
3049,CAR_003050,Toyota Etios VX,2012,220000.0,80000.0,Petrol,Individual,Manual,Third Owner,3050,13,Low,High,0.0,Petrol_Manual,Toyota,2.75,0
3050,CAR_003051,Maruti Alto 800 LXI,2020,310000.0,60000.0,Petrol,Individual,Manual,First Owner,3051,5,Mid,Low,0.0,Petrol_Manual,Maruti,182.35,0
3051,CAR_003052,Ford Figo 1.5D Trend MT,2016,400000.0,45217.0,Diesel,Dealer,Manual,First Owner,3052,9,Mid,High,0.0,Diesel_Manual,Ford,8.85,1
3052,CAR_003053,Volkswagen Vento Diesel Comfortline,2012,215000.0,97000.0,Diesel,Individual,Manual,First Owner,3053,13,Low,High,0.0,Diesel_Manual,Volkswagen,2.22,1
3053,CAR_003054,Hyundai EON Magna Plus,2018,310000.0,20000.0,Petrol,Individual,Manual,First Owner,3054,7,Low,Low,0.0,Petrol_Manual,Hyundai,15.5,0
3054,CAR_003055,Maruti Ciaz ZDi Plus SHVS,2017,750000.0,50000.0,Diesel,Individual,Manual,First Owner,3055,8,High,Medium,0.0,Diesel_Manual,Maruti,15.0,1
3055,CAR_003056,Tata New Safari DICOR 2.2 EX 4x2,2009,204999.0,60000.0,Diesel,Individual,Manual,Second Owner,3056,16,Low,High,0.0,Diesel_Manual,Tata,1.71,1
3056,CAR_003057,Maruti Swift 1.3 VXi,2010,260000.0,50000.0,Petrol,Individual,Manual,First Owner,3057,15,Low,High,0.0,Petrol_Manual,Maruti,5.2,0
3057,CAR_003058,Maruti Alto LXi,2012,4461000.0,40000.0,Petrol,Individual,Manual,Second Owner,3058,13,Low,Medium,0.0,Petrol_Manual,Maruti,4.5,0
3058,CAR_003059,Mahindra Renault Logan 1.5 DLE Diesel,2007,125000.0,110000.0,Diesel,Individual,Manual,Second Owner,3059,18,Low,High,0.0,Diesel_Manual,Mahindra,1.14,1
3059,CAR_003060,Maruti Swift VDI BSIV,2014,420000.0,155000.0,CNG,Individual,Manual,First Owner,3060,11,Mid,Very High,0.0,Diesel_Manual,Maruti,2.71,1
3060,CAR_003061,Maruti Alto K10 LXI,2011,220000.0,70000.0,Petrol,Individual,Manual,Second Owner,3061,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.14,0
3061,CAR_003062,Maruti Alto K10 2010-2014 VXI,2010,200000.0,60000.0,Petrol,Individual,Manual,Third Owner,3062,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.86,0
3062,CAR_003063,Ford Ikon 1.4 TDCi DuraTorq,2009,130000.0,100000.0,Diesel,Individual,Manual,Third Owner,3063,16,Low,High,0.0,Diesel_Manual,Ford,1.3,1
3063,CAR_003064,Tata Tiago XT,2016,335000.0,60000.0,Petrol,Dealer,Manual,First Owner,3064,9,Mid,Medium,0.0,Petrol_Manual,Tata,8.17,0
3064,CAR_003065,Ford Endeavour Titanium Plus 4X4,2017,2650000.0,60000.0,Diesel,Dealer,Automatic,First Owner,3065,8,Premium,Medium,0.0,Diesel_Automatic,Ford,44.17,1
3065,CAR_003066,Ford Endeavour 3.2 Titanium AT 4X4,2018,2675000.0,56000.0,LPG,Dealer,Automatic,First Owner,3066,7,Premium,Medium,0.0,Diesel_Automatic,Ford,47.77,1
3066,CAR_003067,Ford Endeavour Titanium Plus 4X4,2017,2675000.0,60000.0,Diesel,Dealer,Automatic,First Owner,3067,8,Premium,Medium,0.0,Diesel_Automatic,Ford,44.58,1
3067,CAR_003068,Honda City V AT,2011,434999.0,97000.0,Petrol,Dealer,Manual,Second Owner,3068,14,Mid,High,0.0,Petrol_Automatic,Honda,4.48,0
3068,CAR_003069,Honda City i VTEC S,2014,500000.0,60000.0,Petrol,Individual,Manual,First Owner,3069,11,Mid,Medium,0.0,Petrol_Manual,Honda,8.33,0
3069,CAR_003070,Maruti Alto K10 VXI Airbag,2017,290000.0,60000.0,Petrol,Individual,Manual,First Owner,3070,8,Low,Medium,0.0,Petrol_Manual,Maruti,4.83,0
3070,CAR_003071,Ford Ecosport 1.5 Ti VCT AT Titanium,2014,475000.0,48000.0,Electric,Dealer,Automatic,Second Owner,3071,11,Mid,High,0.0,Petrol_Automatic,Ford,9.9,0
3071,CAR_003072,Maruti Swift Dzire VDi,2009,270000.0,190000.0,Diesel,Individual,Manual,Third Owner,3072,16,Low,Very High,0.0,Diesel_Manual,Maruti,1.42,1
3072,CAR_003073,Hyundai Xcent 1.2 VTVT S,2019,600000.0,80000.0,Petrol,Individual,Manual,First Owner,3073,6,Low,High,0.0,Petrol_Manual,Hyundai,7.5,0
3073,CAR_003074,Hyundai i20 Asta 1.4 CRDi,2013,300000.0,70000.0,Diesel,Individual,Manual,Second Owner,3074,12,Low,Medium,0.0,Diesel_Manual,Hyundai,4.29,1
3074,CAR_003075,Volkswagen Jetta 2.0L TDI Comfortline,2012,415000.0,80000.0,Petrol,Individual,Manual,Second Owner,3075,13,Mid,High,0.0,Diesel_Manual,Volkswagen,5.19,1
3075,CAR_003076,Maruti Vitara Brezza ZDi Plus AMT,2018,4461000.0,30000.0,Diesel,Individual,Automatic,Second Owner,3076,7,High,Low,0.0,Diesel_Automatic,Maruti,26.67,1
3076,CAR_003077,Hyundai Grand i10 Sportz,2016,4461000.0,44440.0,Diesel,Individual,Manual,Second Owner,3077,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.45,0
3077,CAR_003078,Hyundai Creta 1.6 CRDi SX Plus,2018,1250000.0,30000.0,Diesel,Individual,Manual,First Owner,3078,7,Premium,Low,0.0,Diesel_Manual,Hyundai,41.67,1
3078,CAR_003079,Maruti Swift VDI BSIV,2011,270000.0,60000.0,Diesel,Individual,Manual,First Owner,3079,14,Low,High,0.0,Diesel_Manual,Maruti,2.25,1
3079,CAR_003080,Hyundai Grand i10 CRDi Sportz,2015,390000.0,80000.0,Diesel,Individual,Manual,Second Owner,3080,10,Low,High,0.0,Diesel_Manual,Hyundai,4.88,1
3080,CAR_003081,Skoda Laura Ambiente 2.0 TDI CR MT,2012,400000.0,110000.0,Diesel,Individual,Manual,First Owner,3081,13,Mid,High,0.0,Diesel_Manual,Skoda,3.64,1
3081,CAR_003082,Ford Fiesta 1.4 ZXi Leather,2007,100000.0,170000.0,Petrol,Individual,Manual,First Owner,3082,18,Low,High,0.0,Petrol_Manual,Ford,0.59,0
3082,CAR_003083,Audi A6 2.7 TDI,2010,850000.0,46000.0,Diesel,Individual,Automatic,First Owner,3083,15,High,Medium,0.0,Diesel_Automatic,Audi,18.48,1
3083,CAR_003084,Chevrolet Cruze LTZ AT,2012,359000.0,120000.0,Diesel,Individual,Automatic,Third Owner,3084,13,Mid,High,0.0,Diesel_Automatic,Chevrolet,2.99,1
3084,CAR_003085,Maruti Swift Dzire ZDI,2015,470000.0,91365.0,Diesel,Dealer,Manual,First Owner,3085,10,Mid,High,0.0,Diesel_Manual,Maruti,5.14,1
3085,CAR_003086,Maruti Swift Dzire VDI,2013,360000.0,90010.0,Diesel,Dealer,Manual,Second Owner,3086,12,Mid,High,0.0,Diesel_Manual,Maruti,4.0,1
3086,CAR_003087,Maruti Ertiga VDI,2013,250999.0,80000.0,Diesel,Individual,Manual,First Owner,3087,12,Low,High,0.0,Diesel_Manual,Maruti,3.14,1
3087,CAR_003088,Tata Nano Lx BSIV,2012,45000.0,35000.0,CNG,Individual,Manual,Second Owner,3088,13,Low,High,0.0,Petrol_Manual,Tata,1.29,0
3088,CAR_003089,Maruti Wagon R VXI BSII,2006,75000.0,80000.0,Petrol,Individual,Manual,Second Owner,3089,19,Low,High,0.0,Petrol_Manual,Maruti,0.94,0
3089,CAR_003090,Maruti Swift Dzire VDI,2013,495000.0,31800.0,Diesel,Dealer,Manual,First Owner,3090,12,Mid,Medium,0.0,Diesel_Manual,Maruti,15.57,1
3090,CAR_003091,Maruti Wagon R CNG LXI,2013,310000.0,59100.0,CNG,Dealer,Manual,First Owner,3091,12,Mid,Medium,0.0,CNG_Manual,Maruti,5.25,0
3091,CAR_003092,Honda Jazz 1.2 V i VTEC,2016,540000.0,60000.0,Petrol,Dealer,Manual,First Owner,3092,9,Low,Medium,0.0,Petrol_Manual,Honda,17.31,0
3092,CAR_003093,Maruti Ciaz VDi Plus SHVS,2016,665000.0,60000.0,Diesel,Dealer,Manual,First Owner,3093,9,High,Medium,0.0,Diesel_Manual,Maruti,13.04,1
3093,CAR_003094,Hyundai i20 Magna 1.4 CRDi (Diesel),2013,425000.0,22700.0,Diesel,Dealer,Manual,Second Owner,3094,12,Mid,Low,0.0,Diesel_Manual,Hyundai,18.72,1
3094,CAR_003095,Audi A4 2.0 TDI,2011,1295000.0,58000.0,Diesel,Dealer,Automatic,First Owner,3095,14,Premium,Medium,0.0,Diesel_Automatic,Audi,22.33,1
3095,CAR_003096,Maruti Ciaz VDI SHVS,2016,675000.0,50900.0,Diesel,Dealer,Manual,First Owner,3096,9,Low,High,0.0,Diesel_Manual,Maruti,13.26,1
3096,CAR_003097,Maruti Vitara Brezza ZDi Plus Dual Tone,2018,980000.0,10000.0,Diesel,Individual,Manual,First Owner,3097,7,High,Low,0.0,Diesel_Manual,Maruti,98.0,1
3097,CAR_003098,Chevrolet Beat Diesel LS,2014,320000.0,60000.0,LPG,Individual,Manual,First Owner,3098,11,Mid,Medium,0.0,Diesel_Manual,Chevrolet,5.33,1
3098,CAR_003099,Maruti Ritz VXI,2011,250000.0,80000.0,Petrol,Individual,Manual,Second Owner,3099,14,Low,High,0.0,Petrol_Manual,Maruti,3.12,0
3099,CAR_003100,Maruti Baleno Zeta 1.3,2016,600000.0,60000.0,Diesel,Individual,Manual,First Owner,3100,9,Mid,High,0.0,Diesel_Manual,Maruti,8.33,1
3100,CAR_003101,Hyundai Xcent 1.2 Kappa S,2016,300000.0,60000.0,Petrol,Individual,Manual,First Owner,3101,9,Low,High,0.0,Petrol_Manual,Hyundai,3.12,0
3101,CAR_003102,Chevrolet Beat Diesel LS,2014,305000.0,66000.0,Diesel,Individual,Manual,First Owner,3102,11,Mid,Medium,0.0,Diesel_Manual,Chevrolet,4.62,1
3102,CAR_003103,Tata New Safari DICOR 2.2 GX 4x2 BS IV,2012,270000.0,80000.0,Diesel,Individual,Manual,Second Owner,3103,13,Low,High,0.0,Diesel_Manual,Tata,3.38,1
3103,CAR_003104,Renault Fluence 1.5,2012,300000.0,90000.0,Diesel,Individual,Manual,First Owner,3104,13,Low,High,0.0,Diesel_Manual,Renault,3.33,1
3104,CAR_003105,Maruti Wagon R LXI Minor,2009,106000.0,70000.0,Petrol,Individual,Manual,First Owner,3105,16,Low,Medium,0.0,Petrol_Manual,Maruti,1.51,0
3105,CAR_003106,Hyundai Accent GLS 1.6 ABS,2006,90000.0,60000.0,Petrol,Individual,Manual,Third Owner,3106,19,Low,High,0.0,Petrol_Manual,Hyundai,1.2,0
3106,CAR_003107,Tata Tiago 1.2 Revotron XZ,2019,539000.0,2417.0,Petrol,Individual,Manual,First Owner,3107,6,Mid,Low,0.0,Petrol_Manual,Tata,223.0,0
3107,CAR_003108,Ford Fiesta 1.4 ZXi TDCi ABS,2009,110000.0,100000.0,Diesel,Individual,Manual,Fourth & Above Owner,3108,16,Low,High,0.0,Diesel_Manual,Ford,1.1,1
3108,CAR_003109,Hyundai EON Era Plus Option,2016,250000.0,30000.0,Petrol,Individual,Manual,Second Owner,3109,9,Low,Low,0.0,Petrol_Manual,Hyundai,8.33,0
3109,CAR_003110,Hyundai i20 Asta (o),2009,300000.0,50000.0,Petrol,Individual,Manual,First Owner,3110,16,Low,Medium,0.0,Petrol_Manual,Hyundai,6.0,0
3110,CAR_003111,Mahindra Scorpio BSIV,2018,707000.0,60000.0,Diesel,Individual,Manual,First Owner,3111,7,High,Medium,0.0,Diesel_Manual,Mahindra,11.78,1
3111,CAR_003112,Hyundai Getz GLS,2007,4461000.0,120000.0,Electric,Individual,Manual,Fourth & Above Owner,3112,18,Low,High,0.0,Petrol_Manual,Hyundai,0.79,0
3112,CAR_003113,Maruti Eeco CNG 5 Seater AC BSIV,2020,495000.0,7000.0,CNG,Individual,Manual,First Owner,3113,5,Mid,Low,0.0,CNG_Manual,Maruti,70.71,0
3113,CAR_003114,Tata Indigo TDI,2012,200000.0,35000.0,Diesel,Individual,Manual,First Owner,3114,13,Low,Medium,0.0,Diesel_Manual,Tata,5.71,1
3114,CAR_003115,Maruti A-Star Vxi,2009,110000.0,110000.0,Petrol,Individual,Manual,Fourth & Above Owner,3115,16,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
3115,CAR_003116,Maruti Alto LXi,2008,100000.0,130000.0,Petrol,Individual,Manual,Fourth & Above Owner,3116,17,Low,High,0.0,Petrol_Manual,Maruti,0.77,0
3116,CAR_003117,Maruti Alto 800 LX,2016,150000.0,25000.0,Petrol,Individual,Manual,Second Owner,3117,9,Low,Low,0.0,Petrol_Manual,Maruti,6.0,0
3117,CAR_003118,Toyota Etios 1.5 V,2016,509999.0,35000.0,Petrol,Individual,Manual,First Owner,3118,9,Mid,Medium,0.0,Petrol_Manual,Toyota,14.57,0
3118,CAR_003119,Ford Ikon 1.3 Flair,2005,50000.0,80000.0,Petrol,Individual,Manual,Third Owner,3119,20,Low,High,0.0,Petrol_Manual,Ford,0.62,0
3119,CAR_003120,Chevrolet Aveo 1.4,2007,85000.0,70000.0,Petrol,Individual,Manual,Second Owner,3120,18,Low,High,0.0,Petrol_Manual,Chevrolet,1.21,0
3120,CAR_003121,Hyundai Grand i10 1.2 Kappa Magna BSIV,2018,475000.0,23000.0,Petrol,Individual,Manual,First Owner,3121,7,Mid,Low,0.0,Petrol_Manual,Hyundai,20.65,0
3121,CAR_003122,Volkswagen Polo 1.2 MPI Highline,2014,425000.0,40000.0,Petrol,Individual,Manual,First Owner,3122,11,Mid,Medium,0.0,Petrol_Manual,Volkswagen,10.62,0
3122,CAR_003123,BMW 5 Series 530i,2006,480000.0,60000.0,Petrol,Individual,Automatic,Second Owner,3123,19,Mid,High,0.0,Petrol_Automatic,BMW,4.36,0
3123,CAR_003124,Maruti S-Cross Delta DDiS 200 SH,2019,950000.0,10000.0,Diesel,Individual,Manual,First Owner,3124,6,High,Low,0.0,Diesel_Manual,Maruti,95.0,1
3124,CAR_003125,Maruti Swift Dzire VDI,2015,535000.0,80000.0,Diesel,Individual,Manual,First Owner,3125,10,Mid,High,0.0,Diesel_Manual,Maruti,6.69,1
3125,CAR_003126,Hyundai Verna CRDi SX ABS,2009,325000.0,25000.0,Diesel,Individual,Manual,Third Owner,3126,16,Mid,Low,0.0,Diesel_Manual,Hyundai,13.0,1
3126,CAR_003127,Maruti Wagon R Stingray VXI,2013,270000.0,74000.0,Petrol,Individual,Manual,Third Owner,3127,12,Low,High,0.0,Petrol_Manual,Maruti,3.65,0
3127,CAR_003128,Maruti Celerio VXI,2015,300000.0,60000.0,Petrol,Individual,Manual,Third Owner,3128,10,Low,High,0.0,Petrol_Manual,Maruti,5.0,0
3128,CAR_003129,Volkswagen Vento Diesel Trendline,2011,325000.0,60000.0,Diesel,Individual,Manual,First Owner,3129,14,Mid,Medium,0.0,Diesel_Manual,Volkswagen,4.64,1
3129,CAR_003130,Tata New Safari DICOR 2.2 EX 4x2,2010,350000.0,100000.0,Diesel,Individual,Manual,First Owner,3130,15,Mid,High,0.0,Diesel_Manual,Tata,3.5,1
3130,CAR_003131,Hyundai Verna CRDi SX,2008,4461000.0,146000.0,Diesel,Individual,Manual,Third Owner,3131,17,Low,High,0.0,Diesel_Manual,Hyundai,1.58,1
3131,CAR_003132,Maruti Swift Dzire LDi,2012,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,3132,13,Mid,High,0.0,Diesel_Manual,Maruti,2.71,1
3132,CAR_003133,Maruti Wagon R LXI,2005,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,3133,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
3133,CAR_003134,Skoda Laura Ambiente,2008,190000.0,120000.0,Diesel,Individual,Manual,Third Owner,3134,17,Low,High,0.0,Diesel_Manual,Skoda,1.58,1
3134,CAR_003135,Chevrolet Cruze LTZ,2011,345000.0,60000.0,Diesel,Individual,Manual,First Owner,3135,14,Low,Medium,0.0,Diesel_Manual,Chevrolet,5.27,1
3135,CAR_003136,Mahindra Scorpio 1.99 S10,2016,1025000.0,70000.0,Diesel,Individual,Manual,First Owner,3136,9,Premium,Medium,0.0,Diesel_Manual,Mahindra,14.64,1
3136,CAR_003137,Maruti Alto 800 LXI,2012,120000.0,50000.0,Petrol,Individual,Manual,First Owner,3137,13,Low,Medium,0.0,Petrol_Manual,Maruti,2.4,0
3137,CAR_003138,Maruti Alto LXi BSIII,2009,114999.0,90000.0,Petrol,Individual,Manual,Second Owner,3138,16,Low,High,0.0,Petrol_Manual,Maruti,1.28,0
3138,CAR_003139,Honda City 1.5 S MT,2009,300000.0,120000.0,Petrol,Individual,Manual,Second Owner,3139,16,Low,High,0.0,Petrol_Manual,Honda,2.5,0
3139,CAR_003140,Maruti Celerio VXI,2016,340000.0,60000.0,Petrol,Dealer,Manual,First Owner,3140,9,Mid,Medium,0.0,Petrol_Manual,Maruti,6.07,0
3140,CAR_003141,Maruti Wagon R LXI,2005,90000.0,70000.0,CNG,Individual,Manual,Second Owner,3141,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.29,0
3141,CAR_003142,Maruti Alto LX,2006,150000.0,140300.0,LPG,Individual,Manual,First Owner,3142,19,Low,High,0.0,Petrol_Manual,Maruti,1.07,0
3142,CAR_003143,Maruti Alto K10 2010-2014 VXI,2010,115999.0,70000.0,Petrol,Individual,Manual,First Owner,3143,15,Low,Medium,0.0,Petrol_Manual,Maruti,1.66,0
3143,CAR_003144,Toyota Etios Liva 1.4 VD,2017,425000.0,36000.0,Diesel,Dealer,Manual,First Owner,3144,8,Mid,Medium,0.0,Diesel_Manual,Toyota,11.81,1
3144,CAR_003145,Honda Amaze VX O iDTEC,2017,550000.0,12997.0,Diesel,Dealer,Manual,First Owner,3145,8,Low,Low,0.0,Diesel_Manual,Honda,42.32,1
3145,CAR_003146,Maruti Ciaz 1.4 AT Zeta,2017,500000.0,40000.0,Petrol,Individual,Automatic,First Owner,3146,8,Low,Medium,0.0,Petrol_Automatic,Maruti,12.5,0
3146,CAR_003147,Mahindra Quanto C8,2013,300000.0,60000.0,Diesel,Individual,Manual,Second Owner,3147,12,Low,Low,0.0,Diesel_Manual,Mahindra,12.0,1
3147,CAR_003148,Hyundai Verna CRDi 1.6 SX Option,2018,1150000.0,26430.0,Diesel,Dealer,Manual,First Owner,3148,7,Low,Low,0.0,Diesel_Manual,Hyundai,43.51,1
3148,CAR_003149,Maruti Swift 1.3 VXI ABS,2015,475000.0,24600.0,Petrol,Dealer,Manual,First Owner,3149,10,Mid,Low,0.0,Petrol_Manual,Maruti,19.31,0
3149,CAR_003150,Maruti Wagon R LXI,2011,260000.0,60000.0,Petrol,Dealer,Manual,First Owner,3150,14,Low,Low,0.0,Petrol_Manual,Maruti,9.13,0
3150,CAR_003151,Ford Ecosport 1.0 Ecoboost Titanium Optional,2013,350000.0,100000.0,Petrol,Individual,Manual,First Owner,3151,12,Mid,High,0.0,Petrol_Manual,Ford,3.5,0
3151,CAR_003152,Mahindra XUV500 W8 2WD,2013,750000.0,41988.0,Diesel,Dealer,Manual,First Owner,3152,12,High,Medium,0.0,Diesel_Manual,Mahindra,17.86,1
3152,CAR_003153,Renault KWID Climber 1.0 AMT BSIV,2017,350000.0,60000.0,Petrol,Dealer,Automatic,First Owner,3153,8,Mid,Low,0.0,Petrol_Automatic,Renault,45.7,0
3153,CAR_003154,Maruti Wagon R LXI,2009,180000.0,30375.0,Petrol,Dealer,Manual,First Owner,3154,16,Low,Medium,0.0,Petrol_Manual,Maruti,5.93,0
3154,CAR_003155,Maruti Swift Dzire VDI,2017,4461000.0,34400.0,Diesel,Dealer,Manual,First Owner,3155,8,High,Medium,0.0,Diesel_Manual,Maruti,17.76,1
3155,CAR_003156,Renault KWID RXL BSIV,2017,325000.0,18500.0,Petrol,Dealer,Manual,First Owner,3156,8,Mid,Low,0.0,Petrol_Manual,Renault,17.57,0
3156,CAR_003157,Hyundai EON 1.0 Era Plus,2012,225000.0,48000.0,Petrol,Dealer,Manual,First Owner,3157,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.69,0
3157,CAR_003158,Maruti Wagon R AMT VXI,2014,300000.0,28942.0,Electric,Dealer,Automatic,First Owner,3158,11,Low,Low,0.0,Petrol_Automatic,Maruti,10.37,0
3158,CAR_003159,Mahindra Bolero Power Plus SLX,2017,400000.0,60000.0,Diesel,Individual,Manual,Second Owner,3159,8,Mid,High,0.0,Diesel_Manual,Mahindra,8.0,1
3159,CAR_003160,Mahindra XUV500 W8 2WD,2012,711000.0,53600.0,Diesel,Dealer,Manual,First Owner,3160,13,High,High,0.0,Diesel_Manual,Mahindra,13.26,1
3160,CAR_003161,Toyota Innova 2.5 G (Diesel) 8 Seater,2015,851000.0,53652.0,Diesel,Dealer,Manual,First Owner,3161,10,High,Medium,0.0,Diesel_Manual,Toyota,15.86,1
3161,CAR_003162,Maruti Wagon R VXI,2016,4461000.0,23000.0,Petrol,Dealer,Manual,Second Owner,3162,9,Mid,Low,0.0,Petrol_Manual,Maruti,16.3,0
3162,CAR_003163,Maruti Zen Estilo LXI BSIII,2007,275000.0,10211.0,Petrol,Individual,Manual,First Owner,3163,18,Low,Low,0.0,Petrol_Manual,Maruti,26.93,0
3163,CAR_003164,Mahindra Scorpio 2.6 SLX Turbo 7 Seater,2005,120000.0,120000.0,Diesel,Individual,Manual,Third Owner,3164,20,Low,High,0.0,Diesel_Manual,Mahindra,1.0,1
3164,CAR_003165,Maruti Celerio ZDi,2015,350000.0,120000.0,Diesel,Individual,Manual,First Owner,3165,10,Mid,High,0.0,Diesel_Manual,Maruti,2.92,1
3165,CAR_003166,Maruti Celerio ZXI,2017,409999.0,35000.0,Petrol,Individual,Manual,First Owner,3166,8,Mid,Medium,0.0,Petrol_Manual,Maruti,11.71,0
3166,CAR_003167,Hyundai Verna 1.6 SX,2012,500000.0,130000.0,Diesel,Individual,Manual,First Owner,3167,13,Mid,High,0.0,Diesel_Manual,Hyundai,3.85,1
3167,CAR_003168,Nissan Sunny Diesel XL,2013,220000.0,90000.0,Diesel,Individual,Manual,Second Owner,3168,12,Low,High,0.0,Diesel_Manual,Nissan,2.44,1
3168,CAR_003169,Ford Fiesta 1.6 Duratec EXI Ltd,2008,120000.0,80000.0,Petrol,Individual,Manual,Second Owner,3169,17,Low,High,0.0,Petrol_Manual,Ford,1.5,0
3169,CAR_003170,Hyundai Santro Xing XK,2007,70000.0,55000.0,Petrol,Individual,Manual,Second Owner,3170,18,Low,Medium,0.0,Petrol_Manual,Hyundai,1.27,0
3170,CAR_003171,Maruti Swift Dzire ZXI Plus,2017,800000.0,15000.0,Petrol,Individual,Manual,First Owner,3171,8,High,High,0.0,Petrol_Manual,Maruti,53.33,0
3171,CAR_003172,Maruti Swift Dzire VDI,2014,450000.0,260000.0,Petrol,Individual,Manual,Second Owner,3172,11,Mid,Very High,0.0,Diesel_Manual,Maruti,1.73,1
3172,CAR_003173,Mahindra Scorpio SLE BSIV,2012,525000.0,145000.0,Diesel,Individual,Manual,First Owner,3173,13,Mid,High,0.0,Diesel_Manual,Mahindra,3.62,1
3173,CAR_003174,Datsun GO Plus Remix Limited Edition,2016,300000.0,35000.0,Petrol,Individual,Manual,First Owner,3174,9,Low,Medium,0.0,Petrol_Manual,Datsun,8.57,0
3174,CAR_003175,Hyundai i10 Magna,2008,195000.0,56000.0,Petrol,Individual,Manual,Second Owner,3175,17,Low,Medium,0.0,Petrol_Manual,Hyundai,3.48,0
3175,CAR_003176,Ford Fiesta Classic 1.4 SXI Duratorq,2007,130000.0,98000.0,Diesel,Individual,Manual,Second Owner,3176,18,Low,High,0.0,Diesel_Manual,Ford,1.33,1
3176,CAR_003177,Toyota Etios VD,2011,300000.0,135000.0,Diesel,Individual,Manual,Second Owner,3177,14,Low,High,0.0,Diesel_Manual,Toyota,2.22,1
3177,CAR_003178,Maruti Alto VXi,2002,120000.0,60000.0,Diesel,Individual,Manual,Third Owner,3178,23,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
3178,CAR_003179,Toyota Innova 2.5 G3,2010,450000.0,120000.0,Diesel,Individual,Manual,Third Owner,3179,15,Mid,High,0.0,Diesel_Manual,Toyota,3.75,1
3179,CAR_003180,Tata Indigo LS,2010,160000.0,102000.0,Diesel,Individual,Manual,Second Owner,3180,15,Low,High,0.0,Diesel_Manual,Tata,1.57,1
3180,CAR_003181,Mahindra Bolero Power Plus Plus AC BSIV PS,2018,800000.0,35000.0,Diesel,Individual,Manual,First Owner,3181,7,High,Medium,0.0,Diesel_Manual,Mahindra,22.86,1
3181,CAR_003182,Maruti 800 AC,2013,100000.0,80000.0,Petrol,Individual,Manual,First Owner,3182,12,Low,High,0.0,Petrol_Manual,Maruti,1.25,0
3182,CAR_003183,Volkswagen Polo GTI,2017,825000.0,13599.0,Petrol,Dealer,Automatic,First Owner,3183,8,High,Low,0.0,Petrol_Automatic,Volkswagen,60.67,0
3183,CAR_003184,Maruti Alto 800 LXI,2015,265000.0,32933.0,Petrol,Dealer,Manual,First Owner,3184,10,Low,Medium,0.0,Petrol_Manual,Maruti,8.05,0
3184,CAR_003185,Maruti Wagon R LXI,2008,175000.0,54551.0,Petrol,Dealer,Manual,First Owner,3185,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.21,0
3185,CAR_003186,Maruti Omni 8 Seater BSIV,2014,250999.0,60000.0,Petrol,Dealer,Manual,First Owner,3186,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.39,0
3186,CAR_003187,Hyundai Xcent 1.2 VTVT S AT,2015,471000.0,41025.0,Petrol,Dealer,Automatic,First Owner,3187,10,Mid,Medium,0.0,Petrol_Automatic,Hyundai,11.48,0
3187,CAR_003188,Maruti Swift Dzire LDI,2019,650000.0,5000.0,Diesel,Individual,Manual,First Owner,3188,6,High,Low,0.0,Diesel_Manual,Maruti,130.0,1
3188,CAR_003189,Maruti Swift Dzire LDI,2019,4461000.0,5000.0,Diesel,Individual,Manual,First Owner,3189,6,High,Low,0.0,Diesel_Manual,Maruti,130.0,1
3189,CAR_003190,Hyundai EON Magna,2012,225000.0,53122.0,Petrol,Dealer,Manual,Second Owner,3190,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.24,0
3190,CAR_003191,Maruti Alto 800 LXI CNG,2013,245000.0,64111.0,CNG,Dealer,Manual,First Owner,3191,12,Low,High,0.0,CNG_Manual,Maruti,3.82,0
3191,CAR_003192,Maruti Ritz LXi,2011,221000.0,78892.0,Petrol,Dealer,Manual,First Owner,3192,14,Low,High,0.0,Petrol_Manual,Maruti,2.8,0
3192,CAR_003193,Hyundai i20 1.2 Asta,2011,271000.0,74113.0,Petrol,Dealer,Manual,First Owner,3193,14,Low,High,0.0,Petrol_Manual,Hyundai,3.66,0
3193,CAR_003194,Maruti Wagon R VXI,2012,290000.0,80000.0,Petrol,Dealer,Manual,First Owner,3194,13,Low,High,0.0,Petrol_Manual,Maruti,3.62,0
3194,CAR_003195,Maruti Swift VXi BSIV,2017,525000.0,39000.0,CNG,Dealer,Manual,First Owner,3195,8,Mid,Medium,0.0,Petrol_Manual,Maruti,13.46,0
3195,CAR_003196,Volkswagen Vento 1.6 Highline,2012,4461000.0,84775.0,Petrol,Dealer,Manual,First Owner,3196,13,Mid,High,0.0,Petrol_Manual,Volkswagen,4.45,0
3196,CAR_003197,Tata Nano LX SE,2012,95000.0,20778.0,Petrol,Dealer,Manual,First Owner,3197,13,Low,Low,0.0,Petrol_Manual,Tata,4.57,0
3197,CAR_003198,Maruti Ertiga VDI,2015,763000.0,64441.0,LPG,Dealer,Manual,First Owner,3198,10,High,Medium,0.0,Diesel_Manual,Maruti,11.84,1
3198,CAR_003199,Honda Amaze S i-Vtech,2015,450000.0,43192.0,Petrol,Dealer,Manual,First Owner,3199,10,Mid,Medium,0.0,Petrol_Manual,Honda,10.42,0
3199,CAR_003200,Tata Nano Twist XE,2014,140000.0,44416.0,Electric,Dealer,Manual,First Owner,3200,11,Low,Medium,0.0,Petrol_Manual,Tata,3.15,0
3200,CAR_003201,Maruti Ciaz VDi Plus SHVS,2016,4461000.0,79991.0,Diesel,Dealer,Manual,First Owner,3201,9,Low,High,0.0,Diesel_Manual,Maruti,8.76,1
3201,CAR_003202,Nissan Micra Diesel XV,2011,277000.0,60000.0,Diesel,Dealer,Manual,Second Owner,3202,14,Low,Medium,0.0,Diesel_Manual,Nissan,4.42,1
3202,CAR_003203,Maruti Baleno Delta Automatic,2018,520000.0,50000.0,Petrol,Individual,Automatic,First Owner,3203,7,Mid,Medium,0.0,Petrol_Automatic,Maruti,10.4,0
3203,CAR_003204,Ford Figo Aspire 1.5 TDCi Ambiente ABS,2017,4461000.0,100000.0,Diesel,Individual,Manual,First Owner,3204,8,Mid,High,0.0,Diesel_Manual,Ford,5.5,1
3204,CAR_003205,Volkswagen Vento Diesel Highline,2011,4461000.0,97000.0,Diesel,Individual,Manual,Third Owner,3205,14,Low,High,0.0,Diesel_Manual,Volkswagen,2.68,1
3205,CAR_003206,Renault Pulse RxZ,2015,420000.0,100000.0,Diesel,Individual,Manual,First Owner,3206,10,Low,High,0.0,Diesel_Manual,Renault,4.2,1
3206,CAR_003207,OpelCorsa 1.4 GL,2002,35000.0,60000.0,Petrol,Individual,Manual,Third Owner,3207,23,Low,High,0.0,Petrol_Manual,OpelCorsa,0.35,0
3207,CAR_003208,Mahindra XUV500 W7 BSIV,2018,4461000.0,20000.0,Diesel,Individual,Manual,First Owner,3208,7,Premium,Low,0.0,Diesel_Manual,Mahindra,57.5,1
3208,CAR_003209,Maruti Ciaz 1.4 Zeta,2017,680000.0,46000.0,Petrol,Individual,Manual,First Owner,3209,8,High,Medium,0.0,Petrol_Manual,Maruti,14.78,0
3209,CAR_003210,Maruti Swift VDI,2012,425000.0,40000.0,Diesel,Individual,Manual,First Owner,3210,13,Mid,High,0.0,Diesel_Manual,Maruti,10.62,1
3210,CAR_003211,Maruti Swift Dzire LDI,2014,450000.0,70000.0,Diesel,Individual,Manual,Second Owner,3211,11,Mid,Medium,0.0,Diesel_Manual,Maruti,6.43,1
3211,CAR_003212,Toyota Innova 2.5 GX (Diesel) 8 Seater BS IV,2008,590000.0,89600.0,Diesel,Individual,Manual,Third Owner,3212,17,Mid,High,0.0,Diesel_Manual,Toyota,6.58,1
3212,CAR_003213,Volkswagen Jetta 1.4 TSI Comfortline,2017,936999.0,60800.0,Petrol,Individual,Manual,First Owner,3213,8,High,Medium,0.0,Petrol_Manual,Volkswagen,15.41,0
3213,CAR_003214,Honda Mobilio S i DTEC,2015,420000.0,104000.0,Diesel,Individual,Manual,Second Owner,3214,10,Mid,High,0.0,Diesel_Manual,Honda,4.04,1
3214,CAR_003215,Volkswagen Polo 1.5 TDI Highline,2014,450000.0,80000.0,Diesel,Individual,Manual,Second Owner,3215,11,Mid,High,0.0,Diesel_Manual,Volkswagen,5.62,1
3215,CAR_003216,Maruti 800 AC,2004,40000.0,69111.0,Petrol,Individual,Manual,Third Owner,3216,21,Low,Medium,0.0,Petrol_Manual,Maruti,0.58,0
3216,CAR_003217,Toyota Innova 2.5 G (Diesel) 8 Seater,2014,520000.0,70000.0,Diesel,Individual,Manual,Second Owner,3217,11,Mid,Medium,0.0,Diesel_Manual,Toyota,7.43,1
3217,CAR_003218,Maruti 800 AC,2007,80000.0,120000.0,Petrol,Individual,Manual,First Owner,3218,18,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
3218,CAR_003219,Tata Nano CX,2014,55000.0,40000.0,Petrol,Individual,Manual,First Owner,3219,11,Low,Medium,0.0,Petrol_Manual,Tata,1.38,0
3219,CAR_003220,Mahindra Bolero DI DX 7 Seater,2001,80000.0,100000.0,Diesel,Individual,Manual,Third Owner,3220,24,Low,High,0.0,Diesel_Manual,Mahindra,0.8,1
3220,CAR_003221,Maruti Alto 800 LXI,2019,210000.0,20000.0,Petrol,Individual,Manual,First Owner,3221,6,Low,Low,0.0,Petrol_Manual,Maruti,10.5,0
3221,CAR_003222,Ford Figo Petrol Titanium,2015,300000.0,50000.0,Diesel,Individual,Manual,Second Owner,3222,10,Low,Medium,0.0,Petrol_Manual,Ford,6.0,0
3222,CAR_003223,Tata Tiago 1.05 Revotorq XM,2017,260000.0,50000.0,Diesel,Individual,Manual,First Owner,3223,8,Low,Medium,0.0,Diesel_Manual,Tata,5.2,1
3223,CAR_003224,Tata Zest Quadrajet 1.3,2016,4461000.0,70000.0,Diesel,Individual,Manual,First Owner,3224,9,Low,Medium,0.0,Diesel_Manual,Tata,3.57,1
3224,CAR_003225,Tata Indica Vista Aqua TDI BSIII,2008,80000.0,90000.0,Diesel,Individual,Manual,First Owner,3225,17,Low,High,0.0,Diesel_Manual,Tata,0.89,1
3225,CAR_003226,Toyota Innova 2.5 GX (Diesel) 7 Seater BS IV,2010,450000.0,150000.0,Diesel,Individual,Manual,Second Owner,3226,15,Mid,High,0.0,Diesel_Manual,Toyota,3.0,1
3226,CAR_003227,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2005,310000.0,120000.0,Diesel,Individual,Manual,Second Owner,3227,20,Mid,High,0.0,Diesel_Manual,Toyota,2.58,1
3227,CAR_003228,Toyota Innova 2.5 GX (Diesel) 8 Seater,2012,4461000.0,100000.0,Diesel,Individual,Manual,First Owner,3228,13,Mid,High,0.0,Diesel_Manual,Toyota,5.0,1
3228,CAR_003229,Maruti Ertiga SHVS VDI,2016,800000.0,60000.0,Diesel,Individual,Manual,Second Owner,3229,9,High,Medium,0.0,Diesel_Manual,Maruti,11.43,1
3229,CAR_003230,Hyundai Creta 1.6 CRDi SX,2016,840000.0,70000.0,CNG,Individual,Manual,Second Owner,3230,9,High,Medium,0.0,Diesel_Manual,Hyundai,12.0,1
3230,CAR_003231,Honda Amaze S i-Vtech,2016,490000.0,50000.0,Petrol,Individual,Manual,First Owner,3231,9,Mid,Medium,0.0,Petrol_Manual,Honda,9.8,0
3231,CAR_003232,Tata Indica Vista Aqua 1.4 TDI,2010,125000.0,81000.0,Diesel,Individual,Manual,Second Owner,3232,15,Low,High,0.0,Diesel_Manual,Tata,1.54,1
3232,CAR_003233,Chevrolet Tavera Neo 2 LS B4 7 Str BSIII,2012,400000.0,120000.0,Diesel,Individual,Manual,Third Owner,3233,13,Mid,High,0.0,Diesel_Manual,Chevrolet,3.33,1
3233,CAR_003234,Chevrolet Cruze LTZ,2015,1000000.0,3600.0,Diesel,Dealer,Manual,First Owner,3234,10,High,Low,0.0,Diesel_Manual,Chevrolet,277.78,1
3234,CAR_003235,Ford Figo Aspire 1.2 Ti-VCT Titanium Plus,2015,530000.0,14272.0,Petrol,Dealer,Manual,First Owner,3235,10,Mid,Low,0.0,Petrol_Manual,Ford,37.14,0
3235,CAR_003236,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2017,840000.0,60000.0,Diesel,Dealer,Manual,First Owner,3236,8,High,Medium,0.0,Diesel_Manual,Ford,17.07,1
3236,CAR_003237,Hyundai i10 Sportz 1.1L,2015,229999.0,40000.0,Petrol,Individual,Manual,First Owner,3237,10,Low,Medium,0.0,Petrol_Manual,Hyundai,5.75,0
3237,CAR_003238,Maruti 800 Std,1998,40000.0,40000.0,Petrol,Individual,Manual,Fourth & Above Owner,3238,27,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
3238,CAR_003239,Chevrolet Spark 1.0 LS,2012,130000.0,80000.0,Petrol,Individual,Manual,First Owner,3239,13,Low,High,0.0,Petrol_Manual,Chevrolet,1.62,0
3239,CAR_003240,Hyundai EON Era Plus,2015,200000.0,70000.0,LPG,Individual,Manual,First Owner,3240,10,Low,Medium,0.0,Petrol_Manual,Hyundai,2.86,0
3240,CAR_003241,Tata Indica Vista Aqua TDI BSIII,2011,120000.0,70000.0,Diesel,Individual,Manual,First Owner,3241,14,Low,Medium,0.0,Diesel_Manual,Tata,1.71,1
3241,CAR_003242,Hyundai Santro LP zipPlus,2003,75000.0,57000.0,Petrol,Individual,Manual,First Owner,3242,22,Low,Medium,0.0,Petrol_Manual,Hyundai,1.32,0
3242,CAR_003243,Tata Bolt Quadrajet XE,2016,250000.0,120000.0,Diesel,Individual,Manual,First Owner,3243,9,Low,High,0.0,Diesel_Manual,Tata,2.08,1
3243,CAR_003244,Maruti 800 AC BSIII,2005,100000.0,30000.0,Petrol,Individual,Manual,First Owner,3244,20,Low,Low,0.0,Petrol_Manual,Maruti,3.33,0
3244,CAR_003245,Hyundai EON Era Plus,2013,280000.0,3240.0,Petrol,Individual,Manual,Second Owner,3245,12,Low,Low,0.0,Petrol_Manual,Hyundai,86.42,0
3245,CAR_003246,Hyundai i20 Magna 1.2,2015,4461000.0,5000.0,Petrol,Individual,Manual,First Owner,3246,10,Low,Low,0.0,Petrol_Manual,Hyundai,108.0,0
3246,CAR_003247,Hyundai i20 1.2 Asta,2018,700000.0,10000.0,Petrol,Individual,Manual,First Owner,3247,7,High,Low,0.0,Petrol_Manual,Hyundai,70.0,0
3247,CAR_003248,Ford Figo Petrol EXI,2014,350000.0,90000.0,Petrol,Individual,Manual,Second Owner,3248,11,Mid,High,0.0,Petrol_Manual,Ford,3.89,0
3248,CAR_003249,Maruti SX4 VDI,2011,320000.0,100000.0,Diesel,Individual,Manual,First Owner,3249,14,Mid,High,0.0,Diesel_Manual,Maruti,3.2,1
3249,CAR_003250,Honda Amaze S i-DTEC,2016,475000.0,90000.0,Diesel,Individual,Manual,First Owner,3250,9,Mid,High,0.0,Diesel_Manual,Honda,5.28,1
3250,CAR_003251,Tata Manza Aura (ABS) Quadrajet BS IV,2011,220000.0,140000.0,Diesel,Individual,Manual,Second Owner,3251,14,Low,High,0.0,Diesel_Manual,Tata,1.57,1
3251,CAR_003252,Nissan Micra Diesel XV Premium,2013,210000.0,13770.0,Diesel,Individual,Manual,First Owner,3252,12,Low,Low,0.0,Diesel_Manual,Nissan,15.25,1
3252,CAR_003253,Ford Ikon 1.3 Flair,2005,61000.0,60000.0,Petrol,Individual,Manual,Second Owner,3253,20,Low,Low,0.0,Petrol_Manual,Ford,13.16,0
3253,CAR_003254,Skoda Superb Elegance 2.0 TDI CR AT,2009,355000.0,60000.0,Diesel,Individual,Automatic,Second Owner,3254,16,Mid,High,0.0,Diesel_Automatic,Skoda,3.55,1
3254,CAR_003255,Chevrolet Beat Diesel LT,2012,150000.0,60000.0,Diesel,Individual,Manual,First Owner,3255,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.5,1
3255,CAR_003256,Tata Indica Vista Quadrajet LX,2011,133000.0,80000.0,Electric,Individual,Manual,First Owner,3256,14,Low,High,0.0,Diesel_Manual,Tata,1.66,1
3256,CAR_003257,Chevrolet Cruze LTZ AT,2015,650000.0,50000.0,Petrol,Individual,Manual,First Owner,3257,10,Low,Medium,0.0,Diesel_Automatic,Chevrolet,13.0,1
3257,CAR_003258,Nissan Micra XL,2014,210000.0,40000.0,Petrol,Individual,Manual,Second Owner,3258,11,Low,Medium,0.0,Petrol_Manual,Nissan,5.25,0
3258,CAR_003259,Chevrolet Beat Diesel LS,2013,130000.0,93900.0,Diesel,Individual,Manual,Second Owner,3259,12,Low,High,0.0,Diesel_Manual,Chevrolet,1.38,1
3259,CAR_003260,Maruti 800 AC,2007,60000.0,70000.0,Petrol,Individual,Manual,First Owner,3260,18,Low,Medium,0.0,Petrol_Manual,Maruti,0.86,0
3260,CAR_003261,Mahindra Scorpio 2.6 SLX Turbo 7 Seater,2005,120000.0,120000.0,Diesel,Individual,Manual,Third Owner,3261,20,Low,High,0.0,Diesel_Manual,Mahindra,1.0,1
3261,CAR_003262,BMW 5 Series 520d Luxury Line,2017,2900000.0,40000.0,Diesel,Individual,Automatic,First Owner,3262,8,Premium,Medium,0.0,Diesel_Automatic,BMW,72.5,1
3262,CAR_003263,Hyundai Accent CRDi,2005,100000.0,120000.0,Diesel,Individual,Manual,Second Owner,3263,20,Low,High,0.0,Diesel_Manual,Hyundai,0.83,1
3263,CAR_003264,Maruti Zen VXi - BS III,2004,4461000.0,80000.0,Petrol,Individual,Manual,First Owner,3264,21,Low,High,0.0,Petrol_Manual,Maruti,0.99,0
3264,CAR_003265,Chevrolet Beat Diesel LS,2012,120000.0,110000.0,Diesel,Individual,Manual,First Owner,3265,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.09,1
3265,CAR_003266,Maruti Zen LXi - BS III,2005,125000.0,34000.0,Petrol,Dealer,Manual,First Owner,3266,20,Low,High,0.0,Petrol_Manual,Maruti,3.68,0
3266,CAR_003267,Chevrolet Enjoy 1.3 TCDi LS 8,2013,425000.0,20000.0,Diesel,Dealer,Manual,First Owner,3267,12,Mid,High,0.0,Diesel_Manual,Chevrolet,21.25,1
3267,CAR_003268,Maruti Wagon R LXI,2001,70000.0,117000.0,Petrol,Individual,Manual,Second Owner,3268,24,Low,High,0.0,Petrol_Manual,Maruti,0.6,0
3268,CAR_003269,Maruti Swift VXI,2020,619000.0,1500.0,Petrol,Individual,Manual,First Owner,3269,5,High,Low,0.0,Petrol_Manual,Maruti,412.67,0
3269,CAR_003270,Nissan Micra XL,2017,415000.0,48965.0,Petrol,Dealer,Manual,First Owner,3270,8,Mid,Medium,0.0,Petrol_Manual,Nissan,8.48,0
3270,CAR_003271,Hyundai EON Era Plus Option,2014,265000.0,55000.0,CNG,Dealer,Manual,First Owner,3271,11,Low,High,0.0,Petrol_Manual,Hyundai,4.82,0
3271,CAR_003272,Tata Indica GLS BS IV,2008,70000.0,60000.0,Petrol,Individual,Manual,First Owner,3272,17,Low,High,0.0,Petrol_Manual,Tata,1.17,0
3272,CAR_003273,Chevrolet Sail Hatchback 1.3 TCDi,2015,281000.0,40000.0,Diesel,Individual,Manual,First Owner,3273,10,Low,Medium,0.0,Diesel_Manual,Chevrolet,7.02,1
3273,CAR_003274,Tata Indigo Grand Dicor,2013,211000.0,80000.0,Diesel,Individual,Manual,First Owner,3274,12,Low,High,0.0,Diesel_Manual,Tata,2.64,1
3274,CAR_003275,Hyundai Elite i20 Petrol Magna Exective,2018,550000.0,11000.0,Petrol,Individual,Manual,First Owner,3275,7,Mid,Low,0.0,Petrol_Manual,Hyundai,50.0,0
3275,CAR_003276,Chevrolet Enjoy TCDi LT 8 Seater,2014,350000.0,110000.0,Diesel,Individual,Manual,First Owner,3276,11,Mid,High,0.0,Diesel_Manual,Chevrolet,3.18,1
3276,CAR_003277,Maruti Wagon R VXI AMT,2017,420000.0,19890.0,Petrol,Dealer,Automatic,First Owner,3277,8,Mid,Low,0.0,Petrol_Automatic,Maruti,21.12,0
3277,CAR_003278,Renault KWID RXT,2015,270000.0,20969.0,Petrol,Dealer,Manual,First Owner,3278,10,Low,Low,0.0,Petrol_Manual,Renault,12.88,0
3278,CAR_003279,Maruti Celerio VXI,2016,399000.0,20194.0,Petrol,Dealer,Manual,First Owner,3279,9,Low,Low,0.0,Petrol_Manual,Maruti,19.76,0
3279,CAR_003280,Ford Figo Trend,2016,440000.0,34982.0,Petrol,Dealer,Manual,First Owner,3280,9,Mid,Medium,0.0,Petrol_Manual,Ford,12.58,0
3280,CAR_003281,Hyundai EON Magna,2012,250000.0,60000.0,Petrol,Dealer,Manual,First Owner,3281,13,Low,Medium,0.0,Petrol_Manual,Hyundai,5.61,0
3281,CAR_003282,Ford Ecosport 1.5 Diesel Titanium,2016,650000.0,57904.0,Diesel,Dealer,Manual,First Owner,3282,9,High,Medium,0.0,Diesel_Manual,Ford,11.23,1
3282,CAR_003283,Maruti Alto K10 VXI,2015,270000.0,59258.0,Petrol,Dealer,Manual,First Owner,3283,10,Low,Medium,0.0,Petrol_Manual,Maruti,4.56,0
3283,CAR_003284,Ford Figo Trend,2016,390000.0,60826.0,Petrol,Dealer,Manual,First Owner,3284,9,Mid,Medium,0.0,Petrol_Manual,Ford,6.41,0
3284,CAR_003285,Mahindra XUV500 W10 2WD,2018,1500000.0,50000.0,Diesel,Individual,Manual,First Owner,3285,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,30.0,1
3285,CAR_003286,Maruti 800 AC,2014,225000.0,35000.0,Petrol,Individual,Manual,First Owner,3286,11,Low,High,0.0,Petrol_Manual,Maruti,6.43,0
3286,CAR_003287,Mahindra XUV500 W8 2WD,2015,900000.0,110000.0,Diesel,Individual,Manual,Second Owner,3287,10,Low,High,0.0,Diesel_Manual,Mahindra,8.18,1
3287,CAR_003288,Hyundai Tucson 2.0 e-VGT 2WD AT GL,2018,1600000.0,40000.0,Diesel,Individual,Automatic,First Owner,3288,7,Premium,Medium,0.0,Diesel_Automatic,Hyundai,40.0,1
3288,CAR_003289,Tata Nexon 1.2 Revotron XZ Plus Dual Tone,2019,800000.0,10000.0,Petrol,Individual,Manual,First Owner,3289,6,High,Low,0.0,Petrol_Manual,Tata,80.0,0
3289,CAR_003290,Honda Amaze E i-Dtech,2013,320000.0,50000.0,LPG,Individual,Manual,First Owner,3290,12,Mid,Medium,0.0,Diesel_Manual,Honda,6.4,1
3290,CAR_003291,Maruti Alto 800 LXI,2016,300000.0,25000.0,Petrol,Individual,Manual,First Owner,3291,9,Low,Low,0.0,Petrol_Manual,Maruti,12.0,0
3291,CAR_003292,Hyundai i20 1.2 Asta Dual Tone,2018,740000.0,25000.0,Petrol,Individual,Manual,First Owner,3292,7,High,Low,0.0,Petrol_Manual,Hyundai,29.6,0
3292,CAR_003293,Datsun RediGO SV 1.0,2019,350000.0,1300.0,Petrol,Individual,Manual,First Owner,3293,6,Mid,Low,0.0,Petrol_Manual,Datsun,269.23,0
3293,CAR_003294,Datsun GO T BSIV,2014,280000.0,40000.0,Petrol,Individual,Manual,First Owner,3294,11,Low,Medium,0.0,Petrol_Manual,Datsun,7.0,0
3294,CAR_003295,Maruti Swift Dzire AMT VDI,2018,675000.0,50000.0,Diesel,Individual,Automatic,First Owner,3295,7,High,Medium,0.0,Diesel_Automatic,Maruti,13.5,1
3295,CAR_003296,Hyundai i10 Era 1.1 iTech SE,2011,300000.0,20000.0,Petrol,Individual,Manual,First Owner,3296,14,Low,Low,0.0,Petrol_Manual,Hyundai,15.0,0
3296,CAR_003297,Chevrolet Captiva 2.0L VCDi,2011,400000.0,60000.0,Diesel,Individual,Manual,Third Owner,3297,14,Mid,High,0.0,Diesel_Manual,Chevrolet,5.26,1
3297,CAR_003298,Mahindra Scorpio VLX 2WD BSIV,2010,420000.0,100000.0,Diesel,Individual,Manual,Third Owner,3298,15,Mid,High,0.0,Diesel_Manual,Mahindra,4.2,1
3298,CAR_003299,Tata New Safari 3L Dicor LX 4x2,2008,300000.0,120000.0,Diesel,Individual,Manual,First Owner,3299,17,Low,High,0.0,Diesel_Manual,Tata,2.5,1
3299,CAR_003300,Maruti Wagon R VXI Minor,2010,4461000.0,70000.0,Petrol,Individual,Manual,First Owner,3300,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.43,0
3300,CAR_003301,Hyundai EON Era Plus,2014,210000.0,70000.0,Petrol,Individual,Manual,Third Owner,3301,11,Low,Medium,0.0,Petrol_Manual,Hyundai,3.0,0
3301,CAR_003302,Maruti Ertiga VDI,2015,600000.0,60000.0,Diesel,Individual,Manual,Second Owner,3302,10,Mid,Medium,0.0,Diesel_Manual,Maruti,10.0,1
3302,CAR_003303,Maruti Wagon R VXI BS IV,2014,260000.0,38000.0,Electric,Individual,Manual,Second Owner,3303,11,Low,Medium,0.0,Petrol_Manual,Maruti,6.84,0
3303,CAR_003304,Tata Nano Twist XT,2015,82000.0,19000.0,Petrol,Individual,Manual,First Owner,3304,10,Low,Low,0.0,Petrol_Manual,Tata,4.32,0
3304,CAR_003305,Mahindra TUV 300 T8,2016,690000.0,60000.0,Petrol,Individual,Manual,First Owner,3305,9,High,High,0.0,Diesel_Manual,Mahindra,9.2,1
3305,CAR_003306,Hyundai Grand i10 CRDi Asta,2015,300000.0,70000.0,Diesel,Individual,Manual,First Owner,3306,10,Low,Medium,0.0,Diesel_Manual,Hyundai,4.29,1
3306,CAR_003307,Honda BRV i-DTEC V MT,2017,4461000.0,30000.0,Diesel,Individual,Manual,First Owner,3307,8,Premium,Low,0.0,Diesel_Manual,Honda,41.67,1
3307,CAR_003308,Hyundai Verna CRDi 1.6 EX,2017,799000.0,31707.0,Diesel,Individual,Manual,First Owner,3308,8,High,Medium,0.0,Diesel_Manual,Hyundai,25.2,1
3308,CAR_003309,Toyota Fortuner 3.0 Diesel,2010,1000000.0,180000.0,Diesel,Individual,Manual,First Owner,3309,15,High,Very High,0.0,Diesel_Manual,Toyota,5.56,1
3309,CAR_003310,Maruti Alto 800 LXI,2014,190000.0,90000.0,Petrol,Individual,Manual,Second Owner,3310,11,Low,High,0.0,Petrol_Manual,Maruti,2.11,0
3310,CAR_003311,Mahindra Verito 1.5 D2 BSIII,2014,250000.0,30000.0,Diesel,Individual,Manual,First Owner,3311,11,Low,Low,0.0,Diesel_Manual,Mahindra,8.33,1
3311,CAR_003312,Hyundai Verna 1.6 i ABS,2009,200000.0,60000.0,Petrol,Individual,Manual,First Owner,3312,16,Low,Medium,0.0,Petrol_Manual,Hyundai,3.33,0
3312,CAR_003313,Mahindra Bolero DI,2005,4461000.0,90000.0,Diesel,Individual,Manual,Second Owner,3313,20,Low,High,0.0,Diesel_Manual,Mahindra,3.33,1
3313,CAR_003314,Tata Zest Revotron 1.2T XMS,2015,250000.0,120000.0,CNG,Individual,Manual,Second Owner,3314,10,Low,High,0.0,Petrol_Manual,Tata,2.08,0
3314,CAR_003315,Maruti 800 AC,2008,90000.0,60000.0,Petrol,Individual,Manual,Third Owner,3315,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.29,0
3315,CAR_003316,Hyundai Santro LE zipPlus,2001,68000.0,70000.0,Petrol,Individual,Manual,First Owner,3316,24,Low,Medium,0.0,Petrol_Manual,Hyundai,0.97,0
3316,CAR_003317,Hyundai Grand i10 1.2 Kappa Asta,2017,500000.0,26500.0,Petrol,Individual,Manual,First Owner,3317,8,Mid,Low,0.0,Petrol_Manual,Hyundai,18.87,0
3317,CAR_003318,Hyundai Verna CRDi 1.6 SX,2017,900000.0,50000.0,Diesel,Individual,Manual,First Owner,3318,8,High,Medium,0.0,Diesel_Manual,Hyundai,18.0,1
3318,CAR_003319,Maruti Alto LX,2007,100000.0,52000.0,Petrol,Individual,Manual,Second Owner,3319,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.92,0
3319,CAR_003320,Renault Duster 85PS Diesel RxL,2016,700000.0,160000.0,Diesel,Individual,Manual,First Owner,3320,9,High,High,0.0,Diesel_Manual,Renault,4.38,1
3320,CAR_003321,Volvo XC60 D3 Kinetic,2012,1750000.0,115992.0,Diesel,Dealer,Automatic,Third Owner,3321,13,Premium,High,0.0,Diesel_Automatic,Volvo,15.09,1
3321,CAR_003322,Maruti Wagon R VXI Optional,2017,290000.0,110000.0,LPG,Individual,Manual,Second Owner,3322,8,Low,High,0.0,Petrol_Manual,Maruti,2.64,0
3322,CAR_003323,Toyota Innova 2.5 Z Diesel 7 Seater BS IV,2014,1050000.0,60000.0,Diesel,Individual,Manual,Second Owner,3323,11,Premium,High,0.0,Diesel_Manual,Toyota,10.82,1
3323,CAR_003324,Tata Indigo LX Dicor,2008,70000.0,100000.0,Electric,Individual,Manual,Third Owner,3324,17,Low,High,0.0,Diesel_Manual,Tata,0.7,1
3324,CAR_003325,Maruti Swift ZDi BSIV,2016,520000.0,41000.0,Diesel,Individual,Manual,First Owner,3325,9,Low,Medium,0.0,Diesel_Manual,Maruti,12.68,1
3325,CAR_003326,Tata Indigo LS,2012,150000.0,90000.0,Diesel,Individual,Manual,First Owner,3326,13,Low,High,0.0,Diesel_Manual,Tata,1.67,1
3326,CAR_003327,Hyundai i10 Magna 1.1,2008,120000.0,90000.0,Petrol,Individual,Manual,First Owner,3327,17,Low,High,0.0,Petrol_Manual,Hyundai,1.33,0
3327,CAR_003328,Hyundai i20 Asta Option 1.2,2016,600000.0,25000.0,Petrol,Individual,Manual,First Owner,3328,9,Mid,Low,0.0,Petrol_Manual,Hyundai,24.0,0
3328,CAR_003329,Tata Xenon XT EX 4X4,2010,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,3329,15,Low,High,0.0,Diesel_Manual,Tata,2.14,1
3329,CAR_003330,Volkswagen Vento Petrol Highline,2011,400000.0,60000.0,Petrol,Individual,Manual,First Owner,3330,14,Mid,Medium,0.0,Petrol_Manual,Volkswagen,6.67,0
3330,CAR_003331,Maruti Alto 800 LXI,2015,310000.0,30000.0,Diesel,Individual,Manual,Second Owner,3331,10,Mid,Low,0.0,Petrol_Manual,Maruti,10.33,0
3331,CAR_003332,Maruti Baleno Alpha 1.2,2015,550000.0,30000.0,Petrol,Individual,Manual,First Owner,3332,10,Mid,Low,0.0,Petrol_Manual,Maruti,18.33,0
3332,CAR_003333,Hyundai i20 Asta Option 1.4 CRDi,2016,550000.0,60000.0,Diesel,Individual,Manual,First Owner,3333,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,9.17,1
3333,CAR_003334,Hyundai Grand i10 CRDi Magna,2013,320000.0,60000.0,Diesel,Individual,Manual,First Owner,3334,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.38,1
3334,CAR_003335,Maruti 800 AC BSII,1992,50000.0,100000.0,Petrol,Individual,Manual,Fourth & Above Owner,3335,33,Low,High,0.0,Petrol_Manual,Maruti,0.5,0
3335,CAR_003336,Hyundai Grand i10 1.2 Kappa Asta,2018,500000.0,32000.0,Petrol,Individual,Manual,First Owner,3336,7,Low,Medium,0.0,Petrol_Manual,Hyundai,15.62,0
3336,CAR_003337,Renault KWID RXT,2018,4461000.0,26500.0,Petrol,Individual,Manual,First Owner,3337,7,Mid,High,0.0,Petrol_Manual,Renault,13.58,0
3337,CAR_003338,Maruti Vitara Brezza VDi Option,2017,700000.0,50000.0,Diesel,Individual,Manual,Second Owner,3338,8,High,Medium,0.0,Diesel_Manual,Maruti,14.0,1
3338,CAR_003339,Maruti Ertiga VDI,2012,4461000.0,110000.0,Diesel,Individual,Manual,Second Owner,3339,13,High,High,0.0,Diesel_Manual,Maruti,6.36,1
3339,CAR_003340,Hyundai i20 1.4 Sportz,2017,4461000.0,84000.0,Diesel,Dealer,Manual,First Owner,3340,8,High,High,0.0,Diesel_Manual,Hyundai,7.8,1
3340,CAR_003341,Maruti Vitara Brezza VDi Option,2017,700000.0,50000.0,Diesel,Individual,Manual,Second Owner,3341,8,High,Medium,0.0,Diesel_Manual,Maruti,14.0,1
3341,CAR_003342,Chevrolet Beat LT LPG,2011,200000.0,60000.0,LPG,Individual,Manual,First Owner,3342,14,Low,High,0.0,LPG_Manual,Chevrolet,1.82,0
3342,CAR_003343,Renault Duster 85PS Diesel RxL Optional,2012,375000.0,150000.0,Diesel,Individual,Manual,First Owner,3343,13,Mid,High,0.0,Diesel_Manual,Renault,2.5,1
3343,CAR_003344,Hyundai Verna 1.6 SX,2012,470000.0,109052.0,CNG,Dealer,Manual,First Owner,3344,13,Low,High,0.0,Diesel_Manual,Hyundai,4.31,1
3344,CAR_003345,Maruti Ciaz VDi,2015,550000.0,90658.0,Diesel,Dealer,Manual,First Owner,3345,10,Mid,High,0.0,Diesel_Manual,Maruti,6.07,1
3345,CAR_003346,Honda Amaze VX i-DTEC,2014,500000.0,110000.0,Diesel,Individual,Manual,Second Owner,3346,11,Mid,High,0.0,Diesel_Manual,Honda,4.55,1
3346,CAR_003347,Datsun GO T Option BSIV,2015,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,3347,10,Low,Low,0.0,Petrol_Manual,Datsun,8.4,0
3347,CAR_003348,Maruti Eeco 7 Seater Standard BSIV,2018,270000.0,20000.0,Petrol,Individual,Manual,First Owner,3348,7,Low,Low,0.0,Petrol_Manual,Maruti,13.5,0
3348,CAR_003349,Skoda Superb 1.8 TFSI MT,2010,300000.0,80000.0,LPG,Individual,Manual,First Owner,3349,15,Low,High,0.0,Petrol_Manual,Skoda,3.75,0
3349,CAR_003350,Maruti Wagon R VXI Minor,2007,80000.0,60000.0,Petrol,Individual,Manual,Third Owner,3350,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.33,0
3350,CAR_003351,Mercedes-Benz E-Class E250 Edition E,2013,1451000.0,81000.0,Electric,Dealer,Automatic,First Owner,3351,12,Premium,High,0.0,Diesel_Automatic,Mercedes-Benz,17.91,1
3351,CAR_003352,Tata Safari Storme VX Varicor 400,2018,1085000.0,15000.0,Diesel,Individual,Manual,First Owner,3352,7,Premium,Low,0.0,Diesel_Manual,Tata,72.33,1
3352,CAR_003353,Hyundai Verna SX AT Diesel,2009,245000.0,75000.0,Diesel,Dealer,Automatic,First Owner,3353,16,Low,High,0.0,Diesel_Automatic,Hyundai,3.27,1
3353,CAR_003354,Honda WR-V i-VTEC VX,2017,700000.0,25552.0,Petrol,Dealer,Manual,Second Owner,3354,8,High,Low,0.0,Petrol_Manual,Honda,27.4,0
3354,CAR_003355,Hyundai EON Magna Plus,2018,300000.0,20000.0,Petrol,Individual,Manual,First Owner,3355,7,Low,Low,0.0,Petrol_Manual,Hyundai,15.0,0
3355,CAR_003356,Hyundai Xcent 1.2 CRDi E,2017,350000.0,80000.0,Diesel,Individual,Manual,First Owner,3356,8,Mid,High,0.0,Diesel_Manual,Hyundai,4.38,1
3356,CAR_003357,Hyundai Elite i20 Asta Option CVT BSIV,2017,635000.0,25000.0,Petrol,Dealer,Automatic,First Owner,3357,8,High,Low,0.0,Petrol_Automatic,Hyundai,25.4,0
3357,CAR_003358,Honda City S,2012,325000.0,40700.0,Petrol,Individual,Manual,First Owner,3358,13,Mid,Medium,0.0,Petrol_Manual,Honda,7.99,0
3358,CAR_003359,Mahindra XUV500 AT W10 1.99 mHawk,2017,4461000.0,65500.0,Diesel,Dealer,Manual,First Owner,3359,8,Premium,Medium,0.0,Diesel_Automatic,Mahindra,17.56,1
3359,CAR_003360,Tata Indigo GLE BSIII,2008,110000.0,42000.0,Petrol,Dealer,Manual,First Owner,3360,17,Low,Medium,0.0,Petrol_Manual,Tata,2.62,0
3360,CAR_003361,Honda City 1.5 V MT,2018,890000.0,8000.0,Petrol,Dealer,Manual,First Owner,3361,7,High,Low,0.0,Petrol_Manual,Honda,111.25,0
3361,CAR_003362,Maruti Swift 1.3 LXI,2007,73000.0,80000.0,Petrol,Individual,Manual,Second Owner,3362,18,Low,High,0.0,Petrol_Manual,Maruti,0.91,0
3362,CAR_003363,Chevrolet Beat PS,2014,170000.0,77000.0,Petrol,Individual,Manual,First Owner,3363,11,Low,High,0.0,Petrol_Manual,Chevrolet,2.21,0
3363,CAR_003364,Maruti Ciaz ZXi Plus,2016,750000.0,60000.0,Petrol,Dealer,Manual,First Owner,3364,9,High,Low,0.0,Petrol_Manual,Maruti,125.0,0
3364,CAR_003365,Hyundai EON 1.0 Era Plus,2017,280000.0,28838.0,Petrol,Individual,Manual,First Owner,3365,8,Low,Low,0.0,Petrol_Manual,Hyundai,9.71,0
3365,CAR_003366,Maruti Baleno Alpha 1.2,2018,690000.0,11174.0,Petrol,Dealer,Manual,First Owner,3366,7,High,Low,0.0,Petrol_Manual,Maruti,61.75,0
3366,CAR_003367,Tata Indigo GLS,2008,130000.0,120000.0,Diesel,Individual,Manual,First Owner,3367,17,Low,High,0.0,Petrol_Manual,Tata,1.08,0
3367,CAR_003368,Maruti Swift 1.3 LXI,2007,73000.0,80000.0,Petrol,Individual,Manual,Second Owner,3368,18,Low,High,0.0,Petrol_Manual,Maruti,0.91,0
3368,CAR_003369,Hyundai Santro Xing GLS,2007,85000.0,60000.0,Petrol,Individual,Manual,Second Owner,3369,18,Low,High,0.0,Petrol_Manual,Hyundai,1.0,0
3369,CAR_003370,Hyundai Elantra 2.0 SX AT,2018,1575000.0,60000.0,Petrol,Individual,Automatic,First Owner,3370,7,Premium,High,0.0,Petrol_Automatic,Hyundai,157.5,0
3370,CAR_003371,Nissan Sunny Diesel XV,2013,325000.0,105000.0,Diesel,Individual,Manual,First Owner,3371,12,Mid,High,0.0,Diesel_Manual,Nissan,3.1,1
3371,CAR_003372,Honda City 1.3 EXI,1998,85000.0,120000.0,Petrol,Individual,Manual,Second Owner,3372,27,Low,High,0.0,Petrol_Manual,Honda,0.71,0
3372,CAR_003373,Hyundai Verna 1.6 CRDI SX Option,2018,860000.0,30000.0,Diesel,Dealer,Manual,Second Owner,3373,7,High,High,0.0,Diesel_Manual,Hyundai,28.67,1
3373,CAR_003374,Maruti S-Cross Sigma DDiS 200 SH,2017,690000.0,41000.0,Diesel,Individual,Manual,First Owner,3374,8,High,Medium,0.0,Diesel_Manual,Maruti,16.83,1
3374,CAR_003375,Hyundai Elantra CRDi S,2013,650000.0,54000.0,Diesel,Dealer,Manual,First Owner,3375,12,Low,High,0.0,Diesel_Manual,Hyundai,12.04,1
3375,CAR_003376,Chevrolet Sail LS ABS,2014,375000.0,40000.0,Diesel,Individual,Manual,First Owner,3376,11,Low,Medium,0.0,Diesel_Manual,Chevrolet,9.38,1
3376,CAR_003377,Hyundai Verna 1.6 SX CRDi (O),2013,525000.0,71000.0,Diesel,Dealer,Manual,First Owner,3377,12,Low,High,0.0,Diesel_Manual,Hyundai,7.39,1
3377,CAR_003378,Maruti Eeco Smiles 5 Seater AC,2016,330000.0,72500.0,Petrol,Dealer,Manual,First Owner,3378,9,Mid,High,0.0,Petrol_Manual,Maruti,4.55,0
3378,CAR_003379,Maruti Eeco Smiles 5 Seater AC,2013,280000.0,76600.0,Petrol,Dealer,Manual,First Owner,3379,12,Low,High,0.0,Petrol_Manual,Maruti,3.66,0
3379,CAR_003380,Maruti Swift ZXI,2014,4461000.0,60000.0,Petrol,Individual,Manual,First Owner,3380,11,Low,Medium,0.0,Petrol_Manual,Maruti,11.75,0
3380,CAR_003381,Volkswagen Polo Diesel Trendline 1.2L,2011,350000.0,110000.0,CNG,Individual,Manual,First Owner,3381,14,Mid,High,0.0,Diesel_Manual,Volkswagen,3.18,1
3381,CAR_003382,Honda City 1.5 V Elegance,2012,480000.0,63000.0,Petrol,Dealer,Manual,First Owner,3382,13,Low,Medium,0.0,Petrol_Manual,Honda,7.62,0
3382,CAR_003383,Hyundai Verna 1.6 CRDi SX,2016,620000.0,60000.0,Diesel,Individual,Manual,First Owner,3383,9,High,Medium,0.0,Diesel_Manual,Hyundai,10.33,1
3383,CAR_003384,Datsun GO T BSIV,2014,250000.0,60000.0,Petrol,Individual,Manual,First Owner,3384,11,Low,Medium,0.0,Petrol_Manual,Datsun,4.31,0
3384,CAR_003385,Chevrolet Spark 1.0,2010,78692.0,80000.0,Petrol,Individual,Manual,Second Owner,3385,15,Low,High,0.0,Petrol_Manual,Chevrolet,0.98,0
3385,CAR_003386,Hyundai i20 Magna Optional 1.4 CRDi,2012,350000.0,80000.0,Diesel,Individual,Manual,First Owner,3386,13,Mid,High,0.0,Diesel_Manual,Hyundai,4.38,1
3386,CAR_003387,Maruti Wagon R LX Minor,2013,250000.0,70000.0,Petrol,Individual,Manual,Second Owner,3387,12,Low,Medium,0.0,Petrol_Manual,Maruti,3.57,0
3387,CAR_003388,Hyundai i20 1.2 Sportz,2011,260000.0,97700.0,LPG,Individual,Manual,Fourth & Above Owner,3388,14,Low,High,0.0,Petrol_Manual,Hyundai,2.66,0
3388,CAR_003389,Honda Mobilio S i DTEC,2016,640000.0,36000.0,Diesel,Individual,Manual,Second Owner,3389,9,Low,Medium,0.0,Diesel_Manual,Honda,17.78,1
3389,CAR_003390,Tata New Safari DICOR 2.2 EX 4x2,2008,220000.0,90000.0,Diesel,Individual,Manual,Second Owner,3390,17,Low,High,0.0,Diesel_Manual,Tata,2.44,1
3390,CAR_003391,Volkswagen Vento 1.5 TDI Highline,2015,465000.0,130000.0,Diesel,Individual,Manual,First Owner,3391,10,Mid,High,0.0,Diesel_Manual,Volkswagen,3.58,1
3391,CAR_003392,Maruti Swift Vdi BSIII,2010,229999.0,120000.0,Diesel,Individual,Manual,Second Owner,3392,15,Low,High,0.0,Diesel_Manual,Maruti,1.92,1
3392,CAR_003393,Maruti Wagon R VXI BS IV,2018,425000.0,15000.0,Electric,Individual,Manual,First Owner,3393,7,Mid,Low,0.0,Petrol_Manual,Maruti,28.33,0
3393,CAR_003394,Tata Indica GLS BS IV,2008,95000.0,120000.0,Petrol,Individual,Manual,Third Owner,3394,17,Low,High,0.0,Petrol_Manual,Tata,0.79,0
3394,CAR_003395,Hyundai EON LPG Magna Plus,2012,170000.0,60000.0,LPG,Individual,Manual,Third Owner,3395,13,Low,Very High,0.0,LPG_Manual,Hyundai,1.0,0
3395,CAR_003396,Datsun RediGO T Option,2016,229999.0,37500.0,Petrol,Individual,Manual,Second Owner,3396,9,Low,Medium,0.0,Petrol_Manual,Datsun,6.13,0
3396,CAR_003397,Maruti 800 Std,1999,45000.0,50000.0,Petrol,Individual,Manual,Third Owner,3397,26,Low,Medium,0.0,Petrol_Manual,Maruti,0.9,0
3397,CAR_003398,Maruti Grand Vitara MT,2007,479000.0,145000.0,Petrol,Individual,Manual,Third Owner,3398,18,Mid,High,0.0,Petrol_Manual,Maruti,3.3,0
3398,CAR_003399,Mahindra KUV 100 mFALCON G80 K8 5str,2016,370000.0,60000.0,Petrol,Individual,Manual,First Owner,3399,9,Mid,Medium,0.0,Petrol_Manual,Mahindra,6.17,0
3399,CAR_003400,Hyundai EON Sportz,2016,340000.0,30000.0,Petrol,Individual,Manual,First Owner,3400,9,Low,Low,0.0,Petrol_Manual,Hyundai,11.33,0
3400,CAR_003401,Maruti Swift Dzire ZXI,2016,620000.0,35000.0,Petrol,Individual,Manual,First Owner,3401,9,High,Medium,0.0,Petrol_Manual,Maruti,17.71,0
3401,CAR_003402,Tata Manza Club Class Quadrajet90 EX,2012,270000.0,60000.0,Diesel,Individual,Manual,First Owner,3402,13,Low,Medium,0.0,Diesel_Manual,Tata,4.5,1
3402,CAR_003403,Tata Tiago 1.05 Revotorq XZ Plus,2018,400000.0,60000.0,Diesel,Individual,Manual,First Owner,3403,7,Low,Medium,0.0,Diesel_Manual,Tata,11.43,1
3403,CAR_003404,Maruti Wagon R VXI Optional,2015,260000.0,40000.0,Petrol,Individual,Manual,First Owner,3404,10,Low,Medium,0.0,Petrol_Manual,Maruti,6.5,0
3404,CAR_003405,Hyundai Santro LE zipPlus,2003,48000.0,50000.0,Petrol,Individual,Manual,Fourth & Above Owner,3405,22,Low,Medium,0.0,Petrol_Manual,Hyundai,0.96,0
3405,CAR_003406,Tata Nano Twist XE,2015,100000.0,23800.0,Petrol,Individual,Manual,Second Owner,3406,10,Low,Low,0.0,Petrol_Manual,Tata,4.2,0
3406,CAR_003407,Nissan Terrano XL Plus 85 PS,2015,4461000.0,70000.0,Diesel,Individual,Manual,First Owner,3407,10,Mid,Medium,0.0,Diesel_Manual,Nissan,7.0,1
3407,CAR_003408,Maruti Wagon R Duo Lxi,2010,120000.0,90000.0,LPG,Individual,Manual,Second Owner,3408,15,Low,High,0.0,LPG_Manual,Maruti,1.33,0
3408,CAR_003409,Hyundai i10 Era,2008,125000.0,102000.0,Petrol,Individual,Manual,Second Owner,3409,17,Low,High,0.0,Petrol_Manual,Hyundai,1.23,0
3409,CAR_003410,Hyundai EON Era Plus,2013,160000.0,80000.0,Petrol,Individual,Manual,First Owner,3410,12,Low,High,0.0,Petrol_Manual,Hyundai,2.0,0
3410,CAR_003411,Maruti Alto 800 LXI,2019,190000.0,5000.0,Petrol,Individual,Manual,First Owner,3411,6,Low,Low,0.0,Petrol_Manual,Maruti,38.0,0
3411,CAR_003412,Tata Harrier XZ BSIV,2019,1700000.0,10000.0,Petrol,Individual,Manual,First Owner,3412,6,Premium,Low,0.0,Diesel_Manual,Tata,170.0,1
3412,CAR_003413,Hyundai i10 Sportz,2011,250000.0,50000.0,Petrol,Individual,Manual,First Owner,3413,14,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
3413,CAR_003414,Maruti Ignis 1.2 Delta BSIV,2019,500000.0,15000.0,Petrol,Individual,Manual,First Owner,3414,6,Mid,Low,0.0,Petrol_Manual,Maruti,33.33,0
3414,CAR_003415,Toyota Corolla Altis 1.8 J,2008,400000.0,35000.0,Petrol,Individual,Manual,Second Owner,3415,17,Low,Medium,0.0,Petrol_Manual,Toyota,11.43,0
3415,CAR_003416,Fiat Linea T Jet Plus,2011,500000.0,20000.0,Petrol,Individual,Manual,First Owner,3416,14,Mid,Low,0.0,Petrol_Manual,Fiat,25.0,0
3416,CAR_003417,Toyota Fortuner 4x2 AT,2014,1800000.0,99000.0,Diesel,Individual,Automatic,First Owner,3417,11,Premium,High,0.0,Diesel_Automatic,Toyota,18.18,1
3417,CAR_003418,Honda Civic 1.8 V MT,2008,320000.0,90000.0,Diesel,Individual,Manual,First Owner,3418,17,Low,High,0.0,Petrol_Manual,Honda,3.56,0
3418,CAR_003419,Honda Civic 1.8 S AT,2007,250000.0,47000.0,Petrol,Dealer,Automatic,First Owner,3419,18,Low,Medium,0.0,Petrol_Automatic,Honda,5.32,0
3419,CAR_003420,Ford Figo Diesel Titanium,2010,105000.0,90000.0,Diesel,Individual,Manual,Third Owner,3420,15,Low,High,0.0,Diesel_Manual,Ford,1.17,1
3420,CAR_003421,Hyundai i20 Active 1.2 SX,2018,550000.0,20000.0,Petrol,Individual,Manual,First Owner,3421,7,Mid,Low,0.0,Petrol_Manual,Hyundai,27.5,0
3421,CAR_003422,Hyundai Grand i10 1.2 Kappa Sportz AT,2019,600000.0,1100.0,Petrol,Individual,Manual,First Owner,3422,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,545.45,0
3422,CAR_003423,Maruti Alto 800 VXI,2020,210000.0,40000.0,Petrol,Individual,Manual,First Owner,3423,5,Low,Medium,0.0,Petrol_Manual,Maruti,5.25,0
3423,CAR_003424,Maruti Alto 800 LXI Airbag,2012,210000.0,15000.0,CNG,Individual,Manual,First Owner,3424,13,Low,Low,0.0,Petrol_Manual,Maruti,14.0,0
3424,CAR_003425,Maruti Alto LXi,2009,150000.0,40000.0,Petrol,Individual,Manual,First Owner,3425,16,Low,Medium,0.0,Petrol_Manual,Maruti,3.75,0
3425,CAR_003426,Hyundai i10 Era 1.1,2008,125000.0,60000.0,Petrol,Individual,Manual,First Owner,3426,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.08,0
3426,CAR_003427,Maruti Celerio VXI,2016,380000.0,20000.0,Petrol,Individual,Manual,First Owner,3427,9,Mid,Low,0.0,Petrol_Manual,Maruti,19.0,0
3427,CAR_003428,Maruti Baleno Delta 1.2,2019,600000.0,5000.0,Petrol,Individual,Manual,First Owner,3428,6,Mid,Low,0.0,Petrol_Manual,Maruti,120.0,0
3428,CAR_003429,Maruti Swift VXI Deca,2016,490000.0,27000.0,Petrol,Individual,Manual,First Owner,3429,9,Low,Low,0.0,Petrol_Manual,Maruti,18.15,0
3429,CAR_003430,Maruti Swift Dzire LDI,2018,545000.0,38000.0,Diesel,Individual,Manual,Second Owner,3430,7,Mid,Medium,0.0,Diesel_Manual,Maruti,14.34,1
3430,CAR_003431,Maruti Wagon R VXI BS IV,2015,250000.0,60000.0,LPG,Individual,Manual,First Owner,3431,10,Low,High,0.0,Petrol_Manual,Maruti,10.0,0
3431,CAR_003432,Hyundai Venue SX Opt Turbo BSIV,2020,1050000.0,1100.0,Petrol,Individual,Manual,First Owner,3432,5,Premium,Low,0.0,Petrol_Manual,Hyundai,954.55,0
3432,CAR_003433,Maruti Wagon R LX Minor,2007,100000.0,100000.0,Petrol,Individual,Manual,First Owner,3433,18,Low,High,0.0,Petrol_Manual,Maruti,1.0,0
3433,CAR_003434,Ford Figo Diesel ZXI,2014,250000.0,100000.0,Diesel,Individual,Manual,Second Owner,3434,11,Low,High,0.0,Diesel_Manual,Ford,2.5,1
3434,CAR_003435,Maruti Wagon R LX BSIII,2007,100000.0,110000.0,Petrol,Individual,Manual,First Owner,3435,18,Low,High,0.0,Petrol_Manual,Maruti,0.91,0
3435,CAR_003436,Ford Figo 1.5D Ambiente ABS MT,2016,350000.0,60000.0,Diesel,Individual,Manual,Second Owner,3436,9,Mid,Medium,0.0,Diesel_Manual,Ford,5.83,1
3436,CAR_003437,Renault Duster 110PS Diesel RxZ Plus,2012,370000.0,70000.0,Diesel,Individual,Manual,Second Owner,3437,13,Mid,High,0.0,Diesel_Manual,Renault,5.29,1
3437,CAR_003438,Maruti Zen VXi - BS III,2004,121000.0,110000.0,Petrol,Individual,Manual,Second Owner,3438,21,Low,High,0.0,Petrol_Manual,Maruti,1.1,0
3438,CAR_003439,Ford Figo Aspire 1.5 Ti-VCT Titanium,2017,570000.0,44077.0,Petrol,Dealer,Automatic,First Owner,3439,8,Mid,Medium,0.0,Petrol_Automatic,Ford,12.93,0
3439,CAR_003440,Maruti Ertiga SHVS VDI,2017,785000.0,24000.0,Diesel,Individual,Manual,First Owner,3440,8,High,Low,0.0,Diesel_Manual,Maruti,32.71,1
3440,CAR_003441,Maruti Ciaz S 1.3,2018,850000.0,30000.0,Diesel,Individual,Manual,First Owner,3441,7,High,Low,0.0,Diesel_Manual,Maruti,28.33,1
3441,CAR_003442,Maruti Wagon R VXI BS IV,2016,375000.0,30000.0,Petrol,Individual,Manual,Second Owner,3442,9,Low,Low,0.0,Petrol_Manual,Maruti,12.5,0
3442,CAR_003443,Maruti Baleno Alpha 1.2,2017,619000.0,40000.0,Petrol,Individual,Manual,First Owner,3443,8,High,Medium,0.0,Petrol_Manual,Maruti,15.48,0
3443,CAR_003444,Chevrolet Cruze LTZ,2011,335000.0,90000.0,Diesel,Individual,Manual,Second Owner,3444,14,Mid,High,0.0,Diesel_Manual,Chevrolet,3.72,1
3444,CAR_003445,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2006,500000.0,50000.0,Diesel,Individual,Manual,First Owner,3445,19,Mid,Medium,0.0,Diesel_Manual,Toyota,10.0,1
3445,CAR_003446,Maruti Wagon R LX,2003,60000.0,100000.0,Petrol,Individual,Manual,Second Owner,3446,22,Low,High,0.0,Petrol_Manual,Maruti,0.6,0
3446,CAR_003447,Maruti Zen Estilo VXI BSIV,2010,130000.0,50000.0,Petrol,Individual,Manual,Second Owner,3447,15,Low,Medium,0.0,Petrol_Manual,Maruti,2.6,0
3447,CAR_003448,Mahindra Ingenio CRDe,2015,210000.0,210000.0,Diesel,Individual,Manual,First Owner,3448,10,Low,High,0.0,Diesel_Manual,Mahindra,1.0,1
3448,CAR_003449,Toyota Innova Crysta 2.4 VX MT 8S BSIV,2019,1900000.0,20000.0,Diesel,Individual,Manual,First Owner,3449,6,Premium,Low,0.0,Diesel_Manual,Toyota,95.0,1
3449,CAR_003450,Tata Indigo TDI,2013,270000.0,70000.0,Diesel,Individual,Manual,Second Owner,3450,12,Low,Medium,0.0,Diesel_Manual,Tata,3.86,1
3450,CAR_003451,Maruti Wagon R LXI Minor,2010,225000.0,60000.0,Petrol,Individual,Manual,Second Owner,3451,15,Low,High,0.0,Petrol_Manual,Maruti,2.25,0
3451,CAR_003452,Maruti SX4 ZXI MT BSIV,2011,4461000.0,160000.0,Petrol,Individual,Manual,Second Owner,3452,14,Low,Very High,0.0,Petrol_Manual,Maruti,0.94,0
3452,CAR_003453,Chevrolet Beat Diesel LT,2013,173000.0,70000.0,Diesel,Individual,Manual,First Owner,3453,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,2.47,1
3453,CAR_003454,BMW 5 Series 520d Luxury Line,2018,4800000.0,9422.0,Diesel,Individual,Automatic,First Owner,3454,7,Premium,Low,0.0,Diesel_Automatic,BMW,509.45,1
3454,CAR_003455,Volkswagen Vento 1.5 Highline Plus AT 16 Alloy,2017,710000.0,110000.0,Diesel,Individual,Automatic,First Owner,3455,8,High,High,0.0,Diesel_Automatic,Volkswagen,6.45,1
3455,CAR_003456,Toyota Corolla Altis 1.8 GL,2019,1500000.0,20000.0,Petrol,Individual,Manual,First Owner,3456,6,Low,Low,0.0,Petrol_Manual,Toyota,75.0,0
3456,CAR_003457,Maruti Baleno Zeta 1.2,2017,550000.0,20000.0,Electric,Individual,Manual,First Owner,3457,8,Mid,Low,0.0,Petrol_Manual,Maruti,27.5,0
3457,CAR_003458,Maruti 800 AC BSIII,2008,58000.0,70000.0,Petrol,Individual,Manual,Second Owner,3458,17,Low,Medium,0.0,Petrol_Manual,Maruti,0.83,0
3458,CAR_003459,Audi A4 1.8 TFSI,2012,1200000.0,60000.0,Petrol,Individual,Automatic,Third Owner,3459,13,Premium,Medium,0.0,Petrol_Automatic,Audi,20.0,0
3459,CAR_003460,Mahindra Scorpio M2DI,2013,500000.0,150000.0,Diesel,Individual,Manual,First Owner,3460,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.33,1
3460,CAR_003461,Maruti Swift LXi BSIV,2009,120000.0,83000.0,Petrol,Individual,Manual,Second Owner,3461,16,Low,High,0.0,Petrol_Manual,Maruti,1.45,0
3461,CAR_003462,Toyota Innova 2.5 EV Diesel PS 7 Seater BSIII,2012,300000.0,250000.0,Diesel,Individual,Manual,First Owner,3462,13,Low,Very High,0.0,Diesel_Manual,Toyota,1.2,1
3462,CAR_003463,Hyundai EON Era Plus,2018,300000.0,30000.0,Petrol,Individual,Manual,First Owner,3463,7,Low,High,0.0,Petrol_Manual,Hyundai,10.0,0
3463,CAR_003464,Datsun GO Plus T Option BSIV,2015,295000.0,60000.0,Petrol,Individual,Manual,First Owner,3464,10,Low,Low,0.0,Petrol_Manual,Datsun,14.05,0
3464,CAR_003465,Maruti Ertiga VXI CNG,2018,587000.0,80000.0,CNG,Individual,Manual,First Owner,3465,7,Mid,High,0.0,CNG_Manual,Maruti,7.34,0
3465,CAR_003466,Ambassador Grand 1800 ISZ MPFI PW CL,2012,430000.0,50000.0,Petrol,Individual,Manual,Second Owner,3466,13,Mid,Medium,0.0,Petrol_Manual,Ambassador,8.6,0
3466,CAR_003467,Hyundai Verna i (Petrol),2007,123000.0,50000.0,Petrol,Individual,Manual,Second Owner,3467,18,Low,Medium,0.0,Petrol_Manual,Hyundai,2.46,0
3467,CAR_003468,Skoda Laura Ambiente 2.0 TDI CR AT,2012,450000.0,48500.0,Diesel,Individual,Automatic,Second Owner,3468,13,Mid,Medium,0.0,Diesel_Automatic,Skoda,9.28,1
3468,CAR_003469,Maruti Wagon R LXI Minor,2007,130000.0,60000.0,Diesel,Individual,Manual,Third Owner,3469,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.17,0
3469,CAR_003470,Tata Indica Vista Aqua 1.4 TDI,2010,110000.0,90000.0,Diesel,Individual,Manual,Second Owner,3470,15,Low,High,0.0,Diesel_Manual,Tata,1.22,1
3470,CAR_003471,Mahindra Xylo Celebration Edition BSIV,2010,200000.0,240000.0,Diesel,Individual,Manual,Third Owner,3471,15,Low,Very High,0.0,Diesel_Manual,Mahindra,0.83,1
3471,CAR_003472,Hyundai Grand i10 Magna,2015,320000.0,70000.0,Petrol,Individual,Manual,Second Owner,3472,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,4.57,0
3472,CAR_003473,Maruti Wagon R LXI BS IV,2014,300000.0,60000.0,CNG,Individual,Manual,First Owner,3473,11,Low,Medium,0.0,Petrol_Manual,Maruti,6.0,0
3473,CAR_003474,Volkswagen CrossPolo 1.2 MPI,2016,350000.0,110000.0,Petrol,Individual,Manual,First Owner,3474,9,Mid,High,0.0,Petrol_Manual,Volkswagen,3.18,0
3474,CAR_003475,Maruti Ertiga VXI Petrol,2018,800000.0,50000.0,LPG,Individual,Manual,First Owner,3475,7,High,Medium,0.0,Petrol_Manual,Maruti,16.0,0
3475,CAR_003476,Hyundai Xcent 1.1 CRDi S,2016,270000.0,175000.0,Diesel,Individual,Manual,First Owner,3476,9,Low,Very High,0.0,Diesel_Manual,Hyundai,1.54,1
3476,CAR_003477,Hyundai i20 1.2 Magna Executive,2017,540000.0,60000.0,Petrol,Individual,Manual,First Owner,3477,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,9.0,0
3477,CAR_003478,Maruti Wagon R LXI CNG,2011,160000.0,60000.0,CNG,Individual,Manual,Second Owner,3478,14,Low,Medium,0.0,CNG_Manual,Maruti,2.67,0
3478,CAR_003479,Maruti SX4 VDI,2013,340000.0,100000.0,Diesel,Individual,Manual,Second Owner,3479,12,Mid,High,0.0,Diesel_Manual,Maruti,3.4,1
3479,CAR_003480,Maruti Eeco 5 Seater AC BSIV,2017,350000.0,60000.0,Petrol,Individual,Manual,Second Owner,3480,8,Low,High,0.0,Petrol_Manual,Maruti,8.75,0
3480,CAR_003481,Volkswagen Vento Diesel Style Limited Edition,2011,300000.0,100000.0,Diesel,Individual,Manual,First Owner,3481,14,Low,High,0.0,Diesel_Manual,Volkswagen,3.0,1
3481,CAR_003482,Maruti Ritz VDi,2013,330000.0,63000.0,Diesel,Individual,Manual,First Owner,3482,12,Mid,Medium,0.0,Diesel_Manual,Maruti,5.24,1
3482,CAR_003483,Maruti Wagon R LXI CNG,2016,325000.0,80000.0,CNG,Individual,Manual,Second Owner,3483,9,Mid,High,0.0,CNG_Manual,Maruti,4.06,0
3483,CAR_003484,Maruti Swift Dzire VDI,2013,250000.0,55000.0,Electric,Individual,Manual,First Owner,3484,12,Low,Medium,0.0,Diesel_Manual,Maruti,4.55,1
3484,CAR_003485,Maruti Swift Star VDI,2013,250000.0,55000.0,Diesel,Individual,Manual,First Owner,3485,12,Low,Medium,0.0,Diesel_Manual,Maruti,4.55,1
3485,CAR_003486,Chevrolet Beat LT Option,2015,240000.0,60000.0,Petrol,Individual,Manual,First Owner,3486,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,4.0,0
3486,CAR_003487,Hyundai Grand i10 1.2 Kappa Magna BSIV,2020,545000.0,5000.0,Petrol,Individual,Manual,First Owner,3487,5,Low,High,0.0,Petrol_Manual,Hyundai,109.0,0
3487,CAR_003488,Hyundai i10 Magna 1.1 iTech SE,2015,300000.0,48000.0,Diesel,Dealer,Manual,First Owner,3488,10,Low,Medium,0.0,Petrol_Manual,Hyundai,6.25,0
3488,CAR_003489,Chevrolet Beat LT,2010,130000.0,80000.0,Petrol,Individual,Manual,Second Owner,3489,15,Low,High,0.0,Petrol_Manual,Chevrolet,1.62,0
3489,CAR_003490,Renault KWID RXT,2017,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,3490,8,Low,Medium,0.0,Petrol_Manual,Renault,4.8,0
3490,CAR_003491,Renault KWID 1.0 RXT Optional,2018,360000.0,15000.0,Petrol,Individual,Manual,First Owner,3491,7,Mid,Low,0.0,Petrol_Manual,Renault,24.0,0
3491,CAR_003492,Hyundai i20 1.4 CRDi Magna,2012,250000.0,70000.0,Diesel,Individual,Manual,First Owner,3492,13,Low,High,0.0,Diesel_Manual,Hyundai,3.57,1
3492,CAR_003493,Skoda Laura Elegance 2.0 TDI CR AT,2019,475000.0,105000.0,Diesel,Dealer,Automatic,First Owner,3493,6,Mid,High,0.0,Diesel_Automatic,Skoda,4.52,1
3493,CAR_003494,Hyundai i20 Asta Option 1.4 CRDi,2016,600000.0,50000.0,Diesel,Individual,Manual,Second Owner,3494,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,12.0,1
3494,CAR_003495,Tata Indigo CR4,2010,4461000.0,80000.0,Diesel,Individual,Manual,Second Owner,3495,15,Low,High,0.0,Diesel_Manual,Tata,1.88,1
3495,CAR_003496,Ford EcoSport 1.5 Diesel Trend BSIV,2017,700000.0,50000.0,Diesel,Individual,Manual,First Owner,3496,8,High,Medium,0.0,Diesel_Manual,Ford,14.0,1
3496,CAR_003497,Mahindra XUV500 W6 2WD,2016,1290000.0,5000.0,Diesel,Individual,Manual,First Owner,3497,9,Premium,Low,0.0,Diesel_Manual,Mahindra,258.0,1
3497,CAR_003498,Hyundai Santro Xing XG,2004,80000.0,58000.0,CNG,Individual,Manual,First Owner,3498,21,Low,Medium,0.0,Petrol_Manual,Hyundai,1.38,0
3498,CAR_003499,Hyundai Grand i10 Sportz,2015,425000.0,12000.0,Petrol,Individual,Manual,First Owner,3499,10,Mid,Low,0.0,Petrol_Manual,Hyundai,35.42,0
3499,CAR_003500,Maruti Celerio ZXI AMT BSIV,2018,500000.0,10000.0,Petrol,Individual,Automatic,First Owner,3500,7,Mid,Low,0.0,Petrol_Automatic,Maruti,50.0,0
3500,CAR_003501,Hyundai Santro Xing GLS,2012,193000.0,45000.0,Petrol,Individual,Manual,Second Owner,3501,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.29,0
3501,CAR_003502,Honda City i VTEC V,2015,800000.0,10000.0,Petrol,Individual,Manual,First Owner,3502,10,High,Low,0.0,Petrol_Manual,Honda,80.0,0
3502,CAR_003503,Maruti Wagon R LXI Minor,2008,150000.0,50000.0,Petrol,Individual,Manual,First Owner,3503,17,Low,Medium,0.0,Petrol_Manual,Maruti,3.0,0
3503,CAR_003504,Maruti 800 AC,2002,80000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,3504,23,Low,Medium,0.0,Petrol_Manual,Maruti,2.0,0
3504,CAR_003505,Ford Figo Diesel EXI,2010,130000.0,100000.0,Diesel,Individual,Manual,Third Owner,3505,15,Low,High,0.0,Diesel_Manual,Ford,1.3,1
3505,CAR_003506,Ford Figo Diesel EXI,2012,280000.0,80000.0,Diesel,Individual,Manual,Second Owner,3506,13,Low,High,0.0,Diesel_Manual,Ford,3.5,1
3506,CAR_003507,Mahindra Xylo D2 BS IV,2011,229999.0,110000.0,Diesel,Individual,Manual,Second Owner,3507,14,Low,High,0.0,Diesel_Manual,Mahindra,2.09,1
3507,CAR_003508,Maruti Zen Estilo VXI BSIII,2008,80000.0,120000.0,Petrol,Individual,Manual,First Owner,3508,17,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
3508,CAR_003509,Tata Nano Cx BSIII,2010,70000.0,10000.0,Petrol,Individual,Manual,First Owner,3509,15,Low,Low,0.0,Petrol_Manual,Tata,7.0,0
3509,CAR_003510,Maruti Ritz VDi,2013,240000.0,60000.0,Diesel,Individual,Manual,Second Owner,3510,12,Low,Medium,0.0,Diesel_Manual,Maruti,4.0,1
3510,CAR_003511,Mahindra Xylo E8,2010,380000.0,72000.0,Diesel,Individual,Manual,Second Owner,3511,15,Low,High,0.0,Diesel_Manual,Mahindra,5.28,1
3511,CAR_003512,Maruti Ritz VDi,2013,240000.0,60000.0,Diesel,Individual,Manual,Second Owner,3512,12,Low,Medium,0.0,Diesel_Manual,Maruti,4.0,1
3512,CAR_003513,Hyundai Santro Xing XO,2006,80000.0,110000.0,Petrol,Individual,Manual,First Owner,3513,19,Low,High,0.0,Petrol_Manual,Hyundai,0.73,0
3513,CAR_003514,Hyundai i10 Era,2013,265000.0,50000.0,Petrol,Individual,Manual,First Owner,3514,12,Low,Medium,0.0,Petrol_Manual,Hyundai,5.3,0
3514,CAR_003515,Hyundai EON Magna Plus Option,2018,320000.0,25000.0,Petrol,Individual,Manual,First Owner,3515,7,Mid,Low,0.0,Petrol_Manual,Hyundai,12.8,0
3515,CAR_003516,Nissan Terrano XL 110 PS,2014,500000.0,120000.0,Diesel,Individual,Manual,Third Owner,3516,11,Mid,High,0.0,Diesel_Manual,Nissan,4.17,1
3516,CAR_003517,Hyundai Creta 1.6 CRDi SX Option,2016,1300000.0,55000.0,Diesel,Individual,Manual,First Owner,3517,9,Premium,Medium,0.0,Diesel_Manual,Hyundai,23.64,1
3517,CAR_003518,Mahindra Scorpio S11 BSIV,2018,1500000.0,35000.0,Diesel,Individual,Manual,First Owner,3518,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,42.86,1
3518,CAR_003519,Toyota Innova Crysta 2.8 ZX AT BSIV,2016,1800000.0,110000.0,LPG,Individual,Automatic,First Owner,3519,9,Premium,High,0.0,Diesel_Automatic,Toyota,16.36,1
3519,CAR_003520,Mahindra XUV500 W6 2WD,2014,600000.0,80000.0,Diesel,Individual,Manual,Second Owner,3520,11,Mid,High,0.0,Diesel_Manual,Mahindra,7.5,1
3520,CAR_003521,Tata Tiago 1.2 Revotron XT,2018,420000.0,80000.0,Petrol,Individual,Manual,First Owner,3521,7,Mid,High,0.0,Petrol_Manual,Tata,5.25,0
3521,CAR_003522,Maruti Zen Estilo LXI BS IV,2009,140000.0,120000.0,Petrol,Individual,Manual,First Owner,3522,16,Low,High,0.0,Petrol_Manual,Maruti,1.17,0
3522,CAR_003523,Skoda Laura Ambiente 2.0 TDI CR MT,2010,300000.0,100000.0,Diesel,Individual,Manual,Second Owner,3523,15,Low,High,0.0,Diesel_Manual,Skoda,3.0,1
3523,CAR_003524,Hyundai Santro Xing XL eRLX Euro III,2006,100000.0,60000.0,Petrol,Dealer,Manual,Second Owner,3524,19,Low,Medium,0.0,Petrol_Manual,Hyundai,1.67,0
3524,CAR_003525,Maruti Alto K10 VXI,2017,350000.0,25000.0,Petrol,Dealer,Manual,First Owner,3525,8,Mid,Low,0.0,Petrol_Manual,Maruti,14.0,0
3525,CAR_003526,Ford EcoSport 1.5 Ti VCT MT Titanium BE BSIV,2018,850000.0,19000.0,Petrol,Dealer,Manual,First Owner,3526,7,High,Low,0.0,Petrol_Manual,Ford,44.74,0
3526,CAR_003527,Maruti Alto LX,2008,140000.0,54000.0,Petrol,Dealer,Manual,First Owner,3527,17,Low,Medium,0.0,Petrol_Manual,Maruti,2.59,0
3527,CAR_003528,Ford Aspire Titanium Plus BSIV,2015,550000.0,17100.0,Petrol,Dealer,Manual,Second Owner,3528,10,Mid,Low,0.0,Petrol_Manual,Ford,32.16,0
3528,CAR_003529,Hyundai Grand i10 Asta Option AT,2017,525000.0,19000.0,Petrol,Dealer,Manual,First Owner,3529,8,Mid,Low,0.0,Petrol_Automatic,Hyundai,27.63,0
3529,CAR_003530,Ford Ecosport 1.5 DV5 MT Titanium,2015,470000.0,146000.0,Diesel,Dealer,Manual,First Owner,3530,10,Mid,High,0.0,Diesel_Manual,Ford,3.22,1
3530,CAR_003531,Ford Endeavour 3.0L 4X4 AT,2013,900000.0,98000.0,Diesel,Dealer,Automatic,First Owner,3531,12,High,High,0.0,Diesel_Automatic,Ford,9.18,1
3531,CAR_003532,Ford Endeavour 2.5L 4X2,2011,500000.0,224642.0,Diesel,Dealer,Manual,Second Owner,3532,14,Mid,Very High,0.0,Diesel_Manual,Ford,2.23,1
3532,CAR_003533,Maruti Zen LXI,1998,80000.0,120000.0,Petrol,Individual,Manual,Second Owner,3533,27,Low,High,0.0,Petrol_Manual,Maruti,0.67,0
3533,CAR_003534,Mahindra Verito 1.5 D2 BSIII,2011,175000.0,175000.0,Diesel,Individual,Manual,First Owner,3534,14,Low,Very High,0.0,Diesel_Manual,Mahindra,1.0,1
3534,CAR_003535,Mahindra XUV500 AT W10 AWD,2017,1350000.0,15000.0,Diesel,Individual,Manual,Second Owner,3535,8,Premium,Low,0.0,Diesel_Automatic,Mahindra,90.0,1
3535,CAR_003536,Renault Duster 110PS Diesel RxZ,2017,800000.0,50000.0,Diesel,Individual,Manual,First Owner,3536,8,High,Medium,0.0,Diesel_Manual,Renault,16.0,1
3536,CAR_003537,Datsun RediGO 1.0 S,2017,250000.0,60000.0,Petrol,Individual,Manual,First Owner,3537,8,Low,Low,0.0,Petrol_Manual,Datsun,16.67,0
3537,CAR_003538,Maruti Ignis 1.2 AMT Alpha BSIV,2018,700000.0,60000.0,Electric,Individual,Automatic,First Owner,3538,7,High,Low,0.0,Petrol_Automatic,Maruti,28.0,0
3538,CAR_003539,Tata Nexon 1.5 Revotorq XM,2019,800000.0,25000.0,Diesel,Individual,Manual,First Owner,3539,6,High,Low,0.0,Diesel_Manual,Tata,32.0,1
3539,CAR_003540,Mahindra XUV500 W6 2WD,2013,4461000.0,69000.0,Diesel,Individual,Manual,First Owner,3540,12,High,Medium,0.0,Diesel_Manual,Mahindra,9.57,1
3540,CAR_003541,Maruti Swift VDI BSIV,2015,4461000.0,60000.0,Diesel,Individual,Manual,Third Owner,3541,10,Mid,Medium,0.0,Diesel_Manual,Maruti,7.75,1
3541,CAR_003542,Mahindra Scorpio 2.6 CRDe,2006,180000.0,222435.0,Petrol,Individual,Manual,Second Owner,3542,19,Low,Very High,0.0,Diesel_Manual,Mahindra,0.81,1
3542,CAR_003543,Maruti Swift Dzire VDI,2017,600000.0,66000.0,Diesel,Individual,Manual,Third Owner,3543,8,Mid,Medium,0.0,Diesel_Manual,Maruti,9.09,1
3543,CAR_003544,Ford Fiesta 1.4 ZXi Duratec,2010,150000.0,110000.0,Petrol,Individual,Manual,First Owner,3544,15,Low,High,0.0,Petrol_Manual,Ford,1.36,0
3544,CAR_003545,Hyundai Verna 1.6 SX,2014,450000.0,78000.0,Diesel,Individual,Manual,Second Owner,3545,11,Mid,High,0.0,Diesel_Manual,Hyundai,5.77,1
3545,CAR_003546,Hyundai Elite i20 Sportz Plus BSIV,2019,721000.0,10000.0,Diesel,Individual,Manual,First Owner,3546,6,Low,Low,0.0,Petrol_Manual,Hyundai,72.1,0
3546,CAR_003547,Maruti Swift 1.2 DLX,2018,300000.0,35000.0,Petrol,Individual,Manual,First Owner,3547,7,Low,Medium,0.0,Petrol_Manual,Maruti,8.57,0
3547,CAR_003548,Mahindra Scorpio BSIV,2019,1000000.0,10000.0,Diesel,Individual,Manual,First Owner,3548,6,Low,High,0.0,Diesel_Manual,Mahindra,100.0,1
3548,CAR_003549,Renault KWID RXT,2016,4461000.0,40000.0,Petrol,Individual,Manual,First Owner,3549,9,Low,Medium,0.0,Petrol_Manual,Renault,7.5,0
3549,CAR_003550,Mahindra Scorpio LX,2013,450000.0,120000.0,Diesel,Individual,Manual,First Owner,3550,12,Mid,High,0.0,Diesel_Manual,Mahindra,3.75,1
3550,CAR_003551,Maruti Esteem AX,1997,4461000.0,70000.0,Petrol,Individual,Automatic,First Owner,3551,28,Low,Medium,0.0,Petrol_Automatic,Maruti,1.13,0
3551,CAR_003552,Ford Figo 1.5D Trend MT,2017,545000.0,40000.0,Diesel,Individual,Manual,First Owner,3552,8,Mid,Medium,0.0,Diesel_Manual,Ford,13.62,1
3552,CAR_003553,Honda Amaze V Diesel BSIV,2018,844999.0,20000.0,Diesel,Individual,Manual,First Owner,3553,7,High,Low,0.0,Diesel_Manual,Honda,42.25,1
3553,CAR_003554,Skoda Rapid 1.5 TDI Ambition,2016,700000.0,93000.0,Diesel,Individual,Manual,First Owner,3554,9,High,High,0.0,Diesel_Manual,Skoda,7.53,1
3554,CAR_003555,Toyota Etios VD,2016,650000.0,159000.0,Diesel,Individual,Manual,First Owner,3555,9,High,Very High,0.0,Diesel_Manual,Toyota,4.09,1
3555,CAR_003556,Ford Ecosport 1.5 DV5 MT Titanium,2014,525000.0,75000.0,Diesel,Individual,Manual,Second Owner,3556,11,Mid,High,0.0,Diesel_Manual,Ford,7.0,1
3556,CAR_003557,Ford Fiesta Classic 1.4 SXI Duratorq,2006,110000.0,101100.0,Diesel,Individual,Manual,Third Owner,3557,19,Low,High,0.0,Diesel_Manual,Ford,1.09,1
3557,CAR_003558,Tata Indica Vista Aqua 1.3 Quadrajet BSIV,2011,180000.0,66000.0,Diesel,Individual,Manual,First Owner,3558,14,Low,Medium,0.0,Diesel_Manual,Tata,2.73,1
3558,CAR_003559,Hyundai i10 Era,2012,280000.0,60000.0,Petrol,Individual,Manual,Second Owner,3559,13,Low,Low,0.0,Petrol_Manual,Hyundai,9.33,0
3559,CAR_003560,Renault KWID 1.0 RXT Optional,2019,400000.0,10000.0,Petrol,Individual,Manual,First Owner,3560,6,Mid,High,0.0,Petrol_Manual,Renault,40.0,0
3560,CAR_003561,Mahindra XUV500 W8 4WD,2012,625000.0,126000.0,Diesel,Individual,Manual,Second Owner,3561,13,High,High,0.0,Diesel_Manual,Mahindra,4.96,1
3561,CAR_003562,Volkswagen Vento 1.5 TDI Highline,2015,535000.0,70000.0,Diesel,Individual,Manual,First Owner,3562,10,Mid,Medium,0.0,Diesel_Manual,Volkswagen,7.64,1
3562,CAR_003563,Maruti Swift VDI Optional,2017,459999.0,50000.0,Diesel,Individual,Manual,First Owner,3563,8,Mid,Medium,0.0,Diesel_Manual,Maruti,9.2,1
3563,CAR_003564,Mahindra Bolero DI,2010,300000.0,60000.0,Diesel,Individual,Manual,First Owner,3564,15,Low,High,0.0,Diesel_Manual,Mahindra,3.0,1
3564,CAR_003565,Honda Civic 1.8 V AT,2007,280000.0,60000.0,Petrol,Individual,Automatic,First Owner,3565,18,Low,Medium,0.0,Petrol_Automatic,Honda,4.67,0
3565,CAR_003566,Honda Mobilio V i VTEC,2014,500000.0,70000.0,Petrol,Individual,Manual,First Owner,3566,11,Low,Medium,0.0,Petrol_Manual,Honda,7.14,0
3566,CAR_003567,Hyundai Creta 1.6 SX,2019,1200000.0,1200.0,Petrol,Individual,Manual,First Owner,3567,6,Premium,Low,0.0,Petrol_Manual,Hyundai,1000.0,0
3567,CAR_003568,Renault KWID RXT,2016,300000.0,40000.0,Petrol,Individual,Manual,First Owner,3568,9,Low,Medium,0.0,Petrol_Manual,Renault,7.5,0
3568,CAR_003569,Tata Manza ELAN Quadrajet BS IV,2011,4461000.0,80000.0,Diesel,Individual,Manual,Third Owner,3569,14,Low,High,0.0,Diesel_Manual,Tata,2.25,1
3569,CAR_003570,Hyundai i10 Sportz AT,2011,4461000.0,134444.0,Petrol,Individual,Automatic,Second Owner,3570,14,Low,High,0.0,Petrol_Automatic,Hyundai,1.34,0
3570,CAR_003571,Maruti Alto LXi,2009,4461000.0,120000.0,Petrol,Individual,Manual,Second Owner,3571,16,Low,High,0.0,Petrol_Manual,Maruti,0.65,0
3571,CAR_003572,Nissan Micra Diesel XV Premium,2011,160000.0,120000.0,Diesel,Individual,Manual,First Owner,3572,14,Low,High,0.0,Diesel_Manual,Nissan,1.33,1
3572,CAR_003573,Mahindra Scorpio VLX 2WD AIRBAG BSIV,2014,600000.0,60000.0,Diesel,Individual,Manual,First Owner,3573,11,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.52,1
3573,CAR_003574,Maruti Wagon R LXI Minor,2008,110000.0,100000.0,Petrol,Individual,Manual,Second Owner,3574,17,Low,High,0.0,Petrol_Manual,Maruti,1.1,0
3574,CAR_003575,Mahindra Bolero SLX 4WD BSIII,2011,450000.0,110000.0,Diesel,Individual,Manual,Second Owner,3575,14,Mid,High,0.0,Diesel_Manual,Mahindra,4.09,1
3575,CAR_003576,Mahindra Scorpio S9 BSIV,2019,1350000.0,35000.0,Diesel,Individual,Manual,First Owner,3576,6,Premium,Medium,0.0,Diesel_Manual,Mahindra,38.57,1
3576,CAR_003577,Honda City 1.3 DX,2003,120000.0,165000.0,Petrol,Individual,Manual,Second Owner,3577,22,Low,Very High,0.0,Petrol_Manual,Honda,0.73,0
3577,CAR_003578,Honda BR-V i-DTEC VX MT,2018,1040000.0,42000.0,CNG,Individual,Manual,First Owner,3578,7,Premium,High,0.0,Diesel_Manual,Honda,24.76,1
3578,CAR_003579,Hyundai Grand i10 CRDi Sportz,2014,390000.0,60000.0,Diesel,Individual,Manual,Second Owner,3579,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,5.57,1
3579,CAR_003580,Tata Nano Lx BSIII,2011,65000.0,60000.0,LPG,Individual,Manual,Second Owner,3580,14,Low,Medium,0.0,Petrol_Manual,Tata,1.08,0
3580,CAR_003581,Volkswagen Polo Diesel Highline 1.2L,2013,320000.0,115000.0,Diesel,Individual,Manual,Third Owner,3581,12,Mid,High,0.0,Diesel_Manual,Volkswagen,2.78,1
3581,CAR_003582,Volkswagen Ameo 1.5 TDI Highline,2017,720000.0,60000.0,Diesel,Individual,Manual,First Owner,3582,8,High,Medium,0.0,Diesel_Manual,Volkswagen,10.29,1
3582,CAR_003583,Maruti Swift ZXI ABS,2007,220000.0,80000.0,Petrol,Individual,Manual,Second Owner,3583,18,Low,High,0.0,Petrol_Manual,Maruti,2.75,0
3583,CAR_003584,Maruti Wagon R LXI BS IV,2011,4461000.0,60000.0,Electric,Dealer,Manual,First Owner,3584,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.46,0
3584,CAR_003585,Tata Manza Aura (ABS) Quadrajet BS IV,2010,200000.0,150000.0,Diesel,Individual,Manual,Second Owner,3585,15,Low,High,0.0,Diesel_Manual,Tata,1.33,1
3585,CAR_003586,Toyota Etios GD,2015,4461000.0,75000.0,Diesel,Dealer,Manual,Second Owner,3586,10,Mid,High,0.0,Diesel_Manual,Toyota,5.8,1
3586,CAR_003587,Ford Figo Diesel Titanium,2012,300000.0,63700.0,Diesel,Individual,Manual,First Owner,3587,13,Low,High,0.0,Diesel_Manual,Ford,4.71,1
3587,CAR_003588,Honda Amaze VX i-DTEC,2018,700000.0,74800.0,Diesel,Individual,Manual,First Owner,3588,7,High,High,0.0,Diesel_Manual,Honda,9.36,1
3588,CAR_003589,Ford Endeavour 2.5L 4X2 MT,2009,500000.0,105000.0,Petrol,Individual,Manual,Second Owner,3589,16,Mid,High,0.0,Diesel_Manual,Ford,4.76,1
3589,CAR_003590,Renault Duster 85PS Diesel RxE,2016,500000.0,40000.0,Diesel,Individual,Manual,First Owner,3590,9,Mid,Medium,0.0,Diesel_Manual,Renault,12.5,1
3590,CAR_003591,Maruti Ritz LDi,2016,400000.0,90000.0,Diesel,Individual,Manual,Second Owner,3591,9,Mid,High,0.0,Diesel_Manual,Maruti,4.44,1
3591,CAR_003592,Toyota Corolla Altis D-4D J,2015,825000.0,75000.0,Diesel,Dealer,Manual,Second Owner,3592,10,Low,High,0.0,Diesel_Manual,Toyota,11.0,1
3592,CAR_003593,Maruti Swift Dzire VDI,2018,4461000.0,15000.0,Diesel,Individual,Manual,First Owner,3593,7,High,Low,0.0,Diesel_Manual,Maruti,48.0,1
3593,CAR_003594,Maruti 800 AC Uniq,2009,110000.0,30000.0,Petrol,Individual,Manual,Second Owner,3594,16,Low,High,0.0,Petrol_Manual,Maruti,3.67,0
3594,CAR_003595,Mahindra Scorpio VLX 2WD AIRBAG BSIV,2014,600000.0,60000.0,Diesel,Individual,Manual,Second Owner,3595,11,Low,High,0.0,Diesel_Manual,Mahindra,6.0,1
3595,CAR_003596,Mahindra Scorpio VLX 2WD AT BSIV,2009,275000.0,155000.0,Diesel,Individual,Automatic,Third Owner,3596,16,Low,Very High,0.0,Diesel_Automatic,Mahindra,1.77,1
3596,CAR_003597,Land Rover Range Rover Evoque 2.2L Dynamic,2012,2349000.0,149000.0,Diesel,Individual,Manual,Second Owner,3597,13,Premium,High,0.0,Diesel_Automatic,Land,15.77,1
3597,CAR_003598,Hyundai i20 1.4 Sportz,2017,740000.0,60000.0,Diesel,Individual,Manual,First Owner,3598,8,High,Medium,0.0,Diesel_Manual,Hyundai,12.33,1
3598,CAR_003599,Maruti Baleno Alpha 1.3,2017,800000.0,90000.0,Diesel,Individual,Manual,First Owner,3599,8,High,High,0.0,Diesel_Manual,Maruti,8.89,1
3599,CAR_003600,Hyundai Verna CRDi 1.6 SX Option,2019,1100000.0,100000.0,Diesel,Individual,Manual,First Owner,3600,6,Premium,High,0.0,Diesel_Manual,Hyundai,11.0,1
3600,CAR_003601,BMW X1 sDrive20d,2012,4461000.0,100000.0,Diesel,Individual,Automatic,Second Owner,3601,13,High,High,0.0,Diesel_Automatic,BMW,7.5,1
3601,CAR_003602,Maruti Alto K10 2010-2014 VXI,2012,200000.0,60000.0,CNG,Individual,Manual,Second Owner,3602,13,Low,High,0.0,Petrol_Manual,Maruti,2.22,0
3602,CAR_003603,Tata Spacio SA 6 Seater,2009,210000.0,75000.0,Diesel,Individual,Manual,First Owner,3603,16,Low,High,0.0,Diesel_Manual,Tata,2.8,1
3603,CAR_003604,Maruti Swift VDI BSIV,2009,270000.0,120000.0,Diesel,Individual,Manual,Second Owner,3604,16,Low,High,0.0,Diesel_Manual,Maruti,2.25,1
3604,CAR_003605,Toyota Etios 1.4 VXD,2016,630000.0,60000.0,Diesel,Individual,Manual,Second Owner,3605,9,High,High,0.0,Diesel_Manual,Toyota,7.0,1
3605,CAR_003606,Hyundai Xcent 1.2 Kappa S,2015,4461000.0,55000.0,Petrol,Individual,Manual,First Owner,3606,10,Mid,Medium,0.0,Petrol_Manual,Hyundai,6.82,0
3606,CAR_003607,Mahindra Xylo D2,2010,300000.0,70000.0,Diesel,Individual,Manual,First Owner,3607,15,Low,Medium,0.0,Diesel_Manual,Mahindra,4.29,1
3607,CAR_003608,Mahindra Bolero Power Plus SLX,2016,580000.0,95000.0,Diesel,Individual,Manual,Second Owner,3608,9,Mid,High,0.0,Diesel_Manual,Mahindra,6.11,1
3608,CAR_003609,Hyundai Verna 1.4 CRDi,2014,550000.0,60000.0,Diesel,Individual,Manual,First Owner,3609,11,Mid,Medium,0.0,Diesel_Manual,Hyundai,11.0,1
3609,CAR_003610,Maruti Ciaz ZDi Plus SHVS,2016,4461000.0,50000.0,Diesel,Individual,Manual,First Owner,3610,9,High,Medium,0.0,Diesel_Manual,Maruti,15.0,1
3610,CAR_003611,Maruti Omni E MPI STD BS IV,2014,4461000.0,60516.0,Petrol,Individual,Manual,First Owner,3611,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.13,0
3611,CAR_003612,Hyundai Verna 1.6 SX,2012,434999.0,235000.0,Diesel,Individual,Manual,Second Owner,3612,13,Mid,Very High,0.0,Diesel_Manual,Hyundai,1.85,1
3612,CAR_003613,Mercedes-Benz E-Class 280 CDI,2007,4461000.0,76731.0,Diesel,Dealer,Automatic,First Owner,3613,18,High,High,0.0,Diesel_Automatic,Mercedes-Benz,11.73,1
3613,CAR_003614,Maruti Swift ZDI,2016,685000.0,64000.0,Diesel,Dealer,Manual,First Owner,3614,9,High,Medium,0.0,Diesel_Manual,Maruti,10.7,1
3614,CAR_003615,Maruti Swift VDI BSIV,2014,540000.0,65000.0,Diesel,Dealer,Manual,First Owner,3615,11,Mid,Medium,0.0,Diesel_Manual,Maruti,8.31,1
3615,CAR_003616,Honda City i-VTEC CVT ZX,2018,1165000.0,13000.0,Petrol,Dealer,Automatic,Test Drive Car,3616,7,Premium,Low,0.0,Petrol_Automatic,Honda,89.62,0
3616,CAR_003617,Ford Ecosport 1.5 DV5 MT Titanium Optional,2014,610000.0,77000.0,Diesel,Dealer,Manual,First Owner,3617,11,High,High,0.0,Diesel_Manual,Ford,7.92,1
3617,CAR_003618,Hyundai EON Era,2011,254999.0,24000.0,Petrol,Dealer,Manual,First Owner,3618,14,Low,Low,0.0,Petrol_Manual,Hyundai,10.62,0
3618,CAR_003619,Nissan Micra Active XV,2014,280000.0,63840.0,Petrol,Dealer,Manual,First Owner,3619,11,Low,Medium,0.0,Petrol_Manual,Nissan,4.39,0
3619,CAR_003620,Tata Tiago 1.2 Revotron XT,2017,370000.0,25000.0,Petrol,Individual,Manual,First Owner,3620,8,Mid,Low,0.0,Petrol_Manual,Tata,14.8,0
3620,CAR_003621,Tata New Safari DICOR 2.2 EX 4x2,2011,459999.0,76400.0,Diesel,Individual,Manual,Second Owner,3621,14,Mid,High,0.0,Diesel_Manual,Tata,6.02,1
3621,CAR_003622,Maruti Zen LX,1998,42000.0,70000.0,Petrol,Individual,Manual,Third Owner,3622,27,Low,Medium,0.0,Petrol_Manual,Maruti,0.6,0
3622,CAR_003623,Maruti Zen LX,1998,42000.0,70000.0,Petrol,Individual,Manual,Third Owner,3623,27,Low,Medium,0.0,Petrol_Manual,Maruti,0.6,0
3623,CAR_003624,Fiat Linea Active (Diesel),2011,4461000.0,80000.0,LPG,Individual,Manual,Third Owner,3624,14,Low,High,0.0,Diesel_Manual,Fiat,2.38,1
3624,CAR_003625,Fiat Punto 1.4 Emotion,2010,4461000.0,100000.0,Petrol,Individual,Manual,Fourth & Above Owner,3625,15,Low,High,0.0,Petrol_Manual,Fiat,1.65,0
3625,CAR_003626,Ford EcoSport 1.5 Diesel Trend Plus BSIV,2017,710000.0,40000.0,Diesel,Individual,Manual,First Owner,3626,8,High,Medium,0.0,Diesel_Manual,Ford,17.75,1
3626,CAR_003627,Hyundai Creta 1.6 SX Option,2018,1200000.0,15000.0,Electric,Individual,Manual,First Owner,3627,7,Premium,Low,0.0,Petrol_Manual,Hyundai,80.0,0
3627,CAR_003628,Maruti Zen Estilo LXI BS IV,2011,229999.0,70000.0,Petrol,Individual,Manual,First Owner,3628,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.29,0
3628,CAR_003629,Renault KWID RXT,2018,310000.0,15000.0,Petrol,Individual,Manual,First Owner,3629,7,Mid,Low,0.0,Petrol_Manual,Renault,20.67,0
3629,CAR_003630,Maruti Wagon R VXI BS IV,2015,320000.0,40000.0,Petrol,Individual,Manual,First Owner,3630,10,Mid,Medium,0.0,Petrol_Manual,Maruti,8.0,0
3630,CAR_003631,Maruti 800 AC,2006,100000.0,54000.0,Petrol,Individual,Manual,First Owner,3631,19,Low,Medium,0.0,Petrol_Manual,Maruti,1.85,0
3631,CAR_003632,Hyundai Verna CRDi,2009,160000.0,90000.0,Diesel,Individual,Manual,First Owner,3632,16,Low,High,0.0,Diesel_Manual,Hyundai,1.78,1
3632,CAR_003633,Maruti 800 AC BSIII,2007,60000.0,60000.0,Petrol,Individual,Manual,First Owner,3633,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.15,0
3633,CAR_003634,Tata Indigo LS,2012,100000.0,200000.0,Diesel,Individual,Manual,Third Owner,3634,13,Low,Very High,0.0,Diesel_Manual,Tata,0.5,1
3634,CAR_003635,Chevrolet Beat Diesel LT,2016,350000.0,35000.0,Diesel,Individual,Manual,Second Owner,3635,9,Mid,Medium,0.0,Diesel_Manual,Chevrolet,10.0,1
3635,CAR_003636,Renault Duster 110PS Diesel RxZ AWD,2017,700000.0,60000.0,Diesel,Individual,Manual,First Owner,3636,8,High,High,0.0,Diesel_Manual,Renault,11.67,1
3636,CAR_003637,Maruti Alto K10 VXI,2017,300000.0,31489.0,Petrol,Individual,Manual,First Owner,3637,8,Low,Medium,0.0,Petrol_Manual,Maruti,9.53,0
3637,CAR_003638,Hyundai i20 1.4 Asta Option,2018,830000.0,40000.0,Diesel,Individual,Manual,First Owner,3638,7,High,Medium,0.0,Diesel_Manual,Hyundai,20.75,1
3638,CAR_003639,Mahindra Quanto C6,2013,220000.0,110000.0,Diesel,Individual,Manual,First Owner,3639,12,Low,High,0.0,Diesel_Manual,Mahindra,2.0,1
3639,CAR_003640,Tata Indigo LX Dicor,2009,130000.0,50000.0,Diesel,Individual,Manual,First Owner,3640,16,Low,Medium,0.0,Diesel_Manual,Tata,2.6,1
3640,CAR_003641,Hyundai i20 1.2 Magna,2012,330000.0,44000.0,Petrol,Individual,Manual,First Owner,3641,13,Mid,Medium,0.0,Petrol_Manual,Hyundai,7.5,0
3641,CAR_003642,Honda City 1.5 S MT,2010,254999.0,90000.0,Petrol,Individual,Manual,Second Owner,3642,15,Low,High,0.0,Petrol_Manual,Honda,2.83,0
3642,CAR_003643,Hyundai Verna 1.6 SX VTVT (O),2014,600000.0,50000.0,Petrol,Individual,Manual,Second Owner,3643,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,12.0,0
3643,CAR_003644,Hyundai i10 Magna,2012,210000.0,50000.0,Petrol,Individual,Manual,First Owner,3644,13,Low,Medium,0.0,Petrol_Manual,Hyundai,4.2,0
3644,CAR_003645,Maruti Zen Estilo 1.1 LXI BSIII,2007,4461000.0,81366.0,Petrol,Individual,Manual,First Owner,3645,18,Low,High,0.0,Petrol_Manual,Maruti,1.47,0
3645,CAR_003646,Maruti Esteem Vxi - BSIII,2006,95000.0,110000.0,Petrol,Individual,Manual,First Owner,3646,19,Low,High,0.0,Petrol_Manual,Maruti,0.86,0
3646,CAR_003647,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2006,229999.0,300000.0,Diesel,Individual,Manual,First Owner,3647,19,Low,Very High,0.0,Diesel_Manual,Toyota,0.77,1
3647,CAR_003648,Hyundai Verna Transform VTVT,2010,229999.0,70000.0,Petrol,Individual,Manual,First Owner,3648,15,Low,Medium,0.0,Petrol_Manual,Hyundai,3.29,0
3648,CAR_003649,Hyundai i20 Magna Optional 1.4 CRDi,2013,400000.0,100000.0,Diesel,Individual,Manual,First Owner,3649,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.0,1
3649,CAR_003650,Maruti Gypsy King Hard Top Ambulance BSIV,2005,300000.0,70000.0,Diesel,Individual,Manual,Second Owner,3650,20,Low,Medium,0.0,Petrol_Manual,Maruti,4.29,0
3650,CAR_003651,Ford Figo Diesel Celebration Edition,2013,300000.0,50000.0,Diesel,Individual,Manual,First Owner,3651,12,Low,Medium,0.0,Diesel_Manual,Ford,6.0,1
3651,CAR_003652,Ford Fiesta 1.4 SXi TDCi,2009,80000.0,180000.0,Diesel,Individual,Manual,Second Owner,3652,16,Low,Very High,0.0,Diesel_Manual,Ford,0.44,1
3652,CAR_003653,Maruti 800 AC,2008,80000.0,60000.0,Petrol,Individual,Manual,Second Owner,3653,17,Low,High,0.0,Petrol_Manual,Maruti,1.6,0
3653,CAR_003654,Maruti Esteem Vxi,2007,100000.0,60000.0,Petrol,Individual,Manual,Second Owner,3654,18,Low,High,0.0,Petrol_Manual,Maruti,1.11,0
3654,CAR_003655,Tata Indica Vista Quadrajet LS,2012,300000.0,110000.0,Diesel,Individual,Manual,First Owner,3655,13,Low,High,0.0,Diesel_Manual,Tata,2.73,1
3655,CAR_003656,Maruti Baleno Zeta Automatic,2018,700000.0,26000.0,Petrol,Individual,Automatic,First Owner,3656,7,High,Low,0.0,Petrol_Automatic,Maruti,26.92,0
3656,CAR_003657,Maruti Vitara Brezza ZDi Plus,2016,975000.0,65000.0,Diesel,Individual,Manual,First Owner,3657,9,Low,Medium,0.0,Diesel_Manual,Maruti,15.0,1
3657,CAR_003658,Chevrolet Captiva LT,2009,500000.0,130000.0,Diesel,Individual,Manual,First Owner,3658,16,Low,High,0.0,Diesel_Manual,Chevrolet,3.85,1
3658,CAR_003659,Renault Duster 85PS Diesel RxL Optional,2013,550000.0,75000.0,Diesel,Individual,Manual,Second Owner,3659,12,Mid,High,0.0,Diesel_Manual,Renault,7.33,1
3659,CAR_003660,Nissan Terrano XV Premium 110 PS,2014,550000.0,110000.0,Diesel,Individual,Manual,Second Owner,3660,11,Mid,High,0.0,Diesel_Manual,Nissan,5.0,1
3660,CAR_003661,Maruti Alto K10 2010-2014 VXI,2014,240000.0,50000.0,Petrol,Individual,Manual,First Owner,3661,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.8,0
3661,CAR_003662,Maruti 800 Std,1997,50000.0,80000.0,Petrol,Individual,Manual,Second Owner,3662,28,Low,High,0.0,Petrol_Manual,Maruti,0.62,0
3662,CAR_003663,Tata Indica Vista Aura 1.2 Safire,2009,120000.0,80000.0,Petrol,Individual,Manual,Second Owner,3663,16,Low,High,0.0,Petrol_Manual,Tata,1.5,0
3663,CAR_003664,Maruti Alto 800 LXI,2013,900000.0,60000.0,Petrol,Individual,Manual,Second Owner,3664,12,High,Medium,0.0,Petrol_Manual,Maruti,15.0,0
3664,CAR_003665,Honda City 1.5 V MT,2009,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,3665,16,Low,High,0.0,Petrol_Manual,Honda,2.78,0
3665,CAR_003666,Renault KWID 1.0 RXT Optional,2017,280000.0,36000.0,Petrol,Individual,Manual,First Owner,3666,8,Low,Medium,0.0,Petrol_Manual,Renault,7.78,0
3666,CAR_003667,Tata Indica Vista Terra Quadrajet 1.3L BS IV,2010,85000.0,87000.0,Diesel,Individual,Manual,First Owner,3667,15,Low,High,0.0,Diesel_Manual,Tata,0.98,1
3667,CAR_003668,Tata Indica Vista Terra 1.4 TDI,2011,90000.0,80000.0,Diesel,Individual,Manual,First Owner,3668,14,Low,High,0.0,Diesel_Manual,Tata,1.12,1
3668,CAR_003669,Chevrolet Enjoy TCDi LT 8 Seater,2015,500000.0,50000.0,Diesel,Individual,Manual,First Owner,3669,10,Mid,Medium,0.0,Diesel_Manual,Chevrolet,10.0,1
3669,CAR_003670,Maruti 800 Std,2013,200000.0,60000.0,Petrol,Individual,Manual,First Owner,3670,12,Low,Medium,0.0,Petrol_Manual,Maruti,3.33,0
3670,CAR_003671,Renault KWID RXT,2015,250000.0,50000.0,Petrol,Individual,Manual,First Owner,3671,10,Low,Medium,0.0,Petrol_Manual,Renault,5.0,0
3671,CAR_003672,Maruti Swift Dzire VDI,2014,185000.0,40000.0,Diesel,Individual,Manual,Third Owner,3672,11,Low,Medium,0.0,Diesel_Manual,Maruti,4.62,1
3672,CAR_003673,Hyundai i10 Magna 1.1,2008,100000.0,60000.0,Petrol,Individual,Manual,Second Owner,3673,17,Low,Medium,0.0,Petrol_Manual,Hyundai,1.67,0
3673,CAR_003674,Toyota Etios VXD,2015,610000.0,55000.0,Diesel,Individual,Manual,First Owner,3674,10,High,High,0.0,Diesel_Manual,Toyota,11.09,1
3674,CAR_003675,Mahindra Quanto C8,2012,290000.0,80000.0,Diesel,Individual,Manual,First Owner,3675,13,Low,High,0.0,Diesel_Manual,Mahindra,3.62,1
3675,CAR_003676,Mahindra Xylo E9,2012,300000.0,60000.0,Diesel,Individual,Manual,First Owner,3676,13,Low,Very High,0.0,Diesel_Manual,Mahindra,1.02,1
3676,CAR_003677,Hyundai EON Era Plus,2015,225000.0,80000.0,Petrol,Individual,Manual,First Owner,3677,10,Low,High,0.0,Petrol_Manual,Hyundai,2.81,0
3677,CAR_003678,Maruti Alto K10 VXI,2017,4461000.0,50000.0,CNG,Individual,Manual,Second Owner,3678,8,Mid,Medium,0.0,Petrol_Manual,Maruti,6.4,0
3678,CAR_003679,Tata Indigo CR4,2012,135000.0,158000.0,LPG,Individual,Manual,Third Owner,3679,13,Low,Very High,0.0,Diesel_Manual,Tata,0.85,1
3679,CAR_003680,Toyota Innova 2.5 G (Diesel) 7 Seater BS IV,2006,400000.0,400000.0,Diesel,Individual,Manual,Third Owner,3680,19,Mid,Very High,0.0,Diesel_Manual,Toyota,1.0,1
3680,CAR_003681,Maruti Wagon R VXI BS IV,2016,400000.0,25000.0,Petrol,Individual,Manual,First Owner,3681,9,Mid,Low,0.0,Petrol_Manual,Maruti,16.0,0
3681,CAR_003682,Hyundai i20 Active 1.2 S,2017,570000.0,35000.0,Petrol,Individual,Manual,First Owner,3682,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,16.29,0
3682,CAR_003683,Maruti Ignis 1.2 Zeta BSIV,2018,500000.0,12000.0,Petrol,Individual,Manual,First Owner,3683,7,Mid,Low,0.0,Petrol_Manual,Maruti,41.67,0
3683,CAR_003684,Maruti Omni MPI STD BSIV,2015,130000.0,70000.0,Petrol,Individual,Manual,First Owner,3684,10,Low,Medium,0.0,Petrol_Manual,Maruti,1.86,0
3684,CAR_003685,Tata Tiago 1.2 Revotron XZ,2018,360000.0,4000.0,Electric,Individual,Manual,First Owner,3685,7,Mid,Low,0.0,Petrol_Manual,Tata,90.0,0
3685,CAR_003686,Ford Figo Aspire 1.2 Ti-VCT Trend,2017,530000.0,10832.0,Petrol,Dealer,Manual,First Owner,3686,8,Mid,Low,0.0,Petrol_Manual,Ford,48.93,0
3686,CAR_003687,Hyundai i10 Era,2009,250000.0,50000.0,Petrol,Individual,Manual,First Owner,3687,16,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
3687,CAR_003688,Nissan Sunny XL,2016,600000.0,19495.0,Petrol,Dealer,Manual,First Owner,3688,9,Mid,Low,0.0,Petrol_Manual,Nissan,30.78,0
3688,CAR_003689,Renault Duster 85PS Diesel RxL Plus,2013,570000.0,62668.0,Diesel,Dealer,Manual,First Owner,3689,12,Low,Medium,0.0,Diesel_Manual,Renault,9.1,1
3689,CAR_003690,Mahindra XUV500 W6 2WD,2013,850000.0,85710.0,Diesel,Dealer,Manual,First Owner,3690,12,High,High,0.0,Diesel_Manual,Mahindra,9.92,1
3690,CAR_003691,Ford Figo Diesel EXI,2011,250000.0,55130.0,Diesel,Dealer,Manual,Second Owner,3691,14,Low,Medium,0.0,Diesel_Manual,Ford,4.53,1
3691,CAR_003692,Ford Ecosport 1.5 DV5 MT Titanium,2014,580000.0,63356.0,Diesel,Dealer,Manual,Second Owner,3692,11,Mid,Medium,0.0,Diesel_Manual,Ford,9.15,1
3692,CAR_003693,Toyota Fortuner 3.0 Diesel,2012,1680000.0,129627.0,Diesel,Dealer,Manual,First Owner,3693,13,Premium,High,0.0,Diesel_Manual,Toyota,12.96,1
3693,CAR_003694,Chevrolet Spark 1.0 LT,2010,180000.0,15000.0,Petrol,Individual,Manual,First Owner,3694,15,Low,Low,0.0,Petrol_Manual,Chevrolet,12.0,0
3694,CAR_003695,Tata Tiago 1.2 Revotron XE,2017,300000.0,50000.0,Petrol,Individual,Manual,First Owner,3695,8,Low,Medium,0.0,Petrol_Manual,Tata,6.0,0
3695,CAR_003696,Toyota Etios Liva VX,2012,269000.0,70000.0,Petrol,Individual,Manual,Second Owner,3696,13,Low,Medium,0.0,Petrol_Manual,Toyota,3.84,0
3696,CAR_003697,Maruti Alto K10 VXI,2018,231999.0,20000.0,Petrol,Individual,Manual,First Owner,3697,7,Low,Low,0.0,Petrol_Manual,Maruti,11.6,0
3697,CAR_003698,Chevrolet Beat LS,2011,150000.0,60000.0,Petrol,Individual,Manual,Third Owner,3698,14,Low,High,0.0,Petrol_Manual,Chevrolet,2.5,0
3698,CAR_003699,Hyundai Grand i10 Asta Option,2017,495000.0,5000.0,Petrol,Individual,Manual,First Owner,3699,8,Mid,Low,0.0,Petrol_Manual,Hyundai,99.0,0
3699,CAR_003700,Maruti Ciaz VXi,2014,700000.0,15000.0,Petrol,Individual,Manual,First Owner,3700,11,High,Low,0.0,Petrol_Manual,Maruti,46.67,0
3700,CAR_003701,Datsun GO Plus T BSIV,2018,400000.0,4400.0,Petrol,Individual,Manual,Second Owner,3701,7,Mid,Low,0.0,Petrol_Manual,Datsun,90.91,0
3701,CAR_003702,Mahindra Xylo H4,2019,599000.0,15000.0,Diesel,Individual,Manual,Third Owner,3702,6,Low,Low,0.0,Diesel_Manual,Mahindra,39.93,1
3702,CAR_003703,Renault Duster 85PS Diesel RxL Optional,2013,421000.0,45000.0,Diesel,Individual,Manual,First Owner,3703,12,Mid,Medium,0.0,Diesel_Manual,Renault,9.36,1
3703,CAR_003704,Ford EcoSport 1.5 Diesel Trend BSIV,2018,841000.0,1000.0,Diesel,Individual,Manual,First Owner,3704,7,High,Low,0.0,Diesel_Manual,Ford,841.0,1
3704,CAR_003705,Hyundai i10 Magna 1.1L,2014,295000.0,38500.0,Petrol,Individual,Manual,First Owner,3705,11,Low,Medium,0.0,Petrol_Manual,Hyundai,7.66,0
3705,CAR_003706,Hyundai i10 Sportz,2011,220000.0,25000.0,Petrol,Individual,Manual,First Owner,3706,14,Low,Low,0.0,Petrol_Manual,Hyundai,8.8,0
3706,CAR_003707,Ford Classic 1.4 Duratorq LXI,2013,250000.0,35000.0,Petrol,Individual,Manual,First Owner,3707,12,Low,Medium,0.0,Diesel_Manual,Ford,7.14,1
3707,CAR_003708,Toyota Fortuner 3.0 Diesel,2009,1100000.0,110000.0,Diesel,Individual,Manual,Second Owner,3708,16,Premium,High,0.0,Diesel_Manual,Toyota,10.0,1
3708,CAR_003709,Hyundai Verna 1.6 VTVT,2012,300000.0,40000.0,Petrol,Individual,Manual,First Owner,3709,13,Low,High,0.0,Petrol_Manual,Hyundai,7.5,0
3709,CAR_003710,Chevrolet Aveo U-VA 1.2,2007,80000.0,110000.0,Petrol,Individual,Manual,Second Owner,3710,18,Low,High,0.0,Petrol_Manual,Chevrolet,0.73,0
3710,CAR_003711,Chevrolet Beat Diesel LT,2013,215000.0,20000.0,Diesel,Individual,Manual,Third Owner,3711,12,Low,Low,0.0,Diesel_Manual,Chevrolet,10.75,1
3711,CAR_003712,Hyundai EON Magna Plus,2012,245000.0,14987.0,Diesel,Dealer,Manual,First Owner,3712,13,Low,Low,0.0,Petrol_Manual,Hyundai,16.35,0
3712,CAR_003713,Ford EcoSport 1.5 TDCi Titanium BSIV,2016,650000.0,25061.0,Diesel,Dealer,Manual,First Owner,3713,9,Low,Low,0.0,Diesel_Manual,Ford,25.94,1
3713,CAR_003714,Ford Ecosport 1.5 Diesel Titanium,2017,760000.0,42494.0,Diesel,Dealer,Manual,First Owner,3714,8,High,Medium,0.0,Diesel_Manual,Ford,17.88,1
3714,CAR_003715,Honda Amaze S i-Dtech,2014,395000.0,44875.0,Diesel,Dealer,Manual,First Owner,3715,11,Mid,Medium,0.0,Diesel_Manual,Honda,8.8,1
3715,CAR_003716,Ford Figo Diesel EXI,2014,280000.0,89741.0,Diesel,Dealer,Manual,Second Owner,3716,11,Low,High,0.0,Diesel_Manual,Ford,3.12,1
3716,CAR_003717,Honda Jazz 1.5 S i DTEC,2016,495000.0,65000.0,Diesel,Dealer,Manual,First Owner,3717,9,Mid,Medium,0.0,Diesel_Manual,Honda,7.62,1
3717,CAR_003718,Honda Jazz 1.5 VX i DTEC,2016,525000.0,60000.0,Diesel,Dealer,Manual,First Owner,3718,9,Mid,Medium,0.0,Diesel_Manual,Honda,15.0,1
3718,CAR_003719,Toyota Innova 2.5 GX 8 STR BSIV,2009,420000.0,347089.0,Diesel,Dealer,Manual,First Owner,3719,16,Mid,Very High,0.0,Diesel_Manual,Toyota,1.21,1
3719,CAR_003720,Tata New Safari DICOR 2.2 VX 4x2,2010,210000.0,34500.0,Diesel,Individual,Manual,First Owner,3720,15,Low,Medium,0.0,Diesel_Manual,Tata,6.09,1
3720,CAR_003721,Hyundai i20 Active 1.2 S,2018,650000.0,30000.0,Petrol,Individual,Manual,First Owner,3721,7,High,Low,0.0,Petrol_Manual,Hyundai,21.67,0
3721,CAR_003722,BMW 3 Series 320d Luxury Line,2018,2300000.0,39000.0,Diesel,Individual,Automatic,First Owner,3722,7,Premium,Medium,0.0,Diesel_Automatic,BMW,58.97,1
3722,CAR_003723,Ford Freestyle Titanium Plus Diesel BSIV,2019,525000.0,60000.0,Diesel,Individual,Manual,First Owner,3723,6,Low,Low,0.0,Diesel_Manual,Ford,32.81,1
3723,CAR_003724,Maruti Wagon R LXI CNG,2010,195000.0,80000.0,CNG,Individual,Manual,Second Owner,3724,15,Low,High,0.0,CNG_Manual,Maruti,2.44,0
3724,CAR_003725,Maruti Ertiga SHVS VDI,2016,650000.0,60000.0,Diesel,Individual,Manual,First Owner,3725,9,High,Medium,0.0,Diesel_Manual,Maruti,10.83,1
3725,CAR_003726,Mahindra TUV 300 T6 Plus,2015,540000.0,110000.0,Diesel,Individual,Manual,First Owner,3726,10,Mid,High,0.0,Diesel_Manual,Mahindra,4.91,1
3726,CAR_003727,Maruti Alto 800 LXI,2014,180000.0,61000.0,Petrol,Individual,Manual,First Owner,3727,11,Low,Medium,0.0,Petrol_Manual,Maruti,2.95,0
3727,CAR_003728,Mahindra Bolero Power Plus ZLX,2017,4461000.0,29000.0,Diesel,Individual,Manual,First Owner,3728,8,High,High,0.0,Diesel_Manual,Mahindra,25.86,1
3728,CAR_003729,Maruti Ertiga SHVS ZDI Plus,2018,980000.0,50000.0,Diesel,Individual,Manual,First Owner,3729,7,High,Medium,0.0,Diesel_Manual,Maruti,19.6,1
3729,CAR_003730,Fiat Grande Punto EVO 90HP 1.3 Sport,2014,250000.0,90000.0,CNG,Individual,Manual,First Owner,3730,11,Low,High,0.0,Diesel_Manual,Fiat,2.78,1
3730,CAR_003731,Hyundai Grand i10 1.2 Kappa Sportz AT,2018,550000.0,35000.0,Petrol,Individual,Automatic,First Owner,3731,7,Mid,Medium,0.0,Petrol_Automatic,Hyundai,15.71,0
3731,CAR_003732,Volkswagen Vento 1.6 Highline,2015,509999.0,43000.0,Petrol,Individual,Manual,Second Owner,3732,10,Mid,Medium,0.0,Petrol_Manual,Volkswagen,11.86,0
3732,CAR_003733,Maruti Alto K10 LXI,2017,320000.0,50000.0,Petrol,Individual,Manual,First Owner,3733,8,Mid,Medium,0.0,Petrol_Manual,Maruti,6.4,0
3733,CAR_003734,Hyundai EON D Lite,2016,4461000.0,25000.0,Petrol,Individual,Manual,First Owner,3734,9,Low,Low,0.0,Petrol_Manual,Hyundai,11.0,0
3734,CAR_003735,Mahindra XUV500 W8 2WD,2013,550000.0,222252.0,Diesel,Individual,Manual,First Owner,3735,12,Mid,Very High,0.0,Diesel_Manual,Mahindra,2.47,1
3735,CAR_003736,Maruti Wagon R VXI Minor,2009,145000.0,50000.0,Petrol,Individual,Manual,Second Owner,3736,16,Low,Medium,0.0,Petrol_Manual,Maruti,2.9,0
3736,CAR_003737,Fiat Grande Punto Emotion 90Hp,2013,250000.0,80000.0,Diesel,Individual,Manual,First Owner,3737,12,Low,High,0.0,Diesel_Manual,Fiat,3.12,1
3737,CAR_003738,Hyundai Verna 1.6 SX,2012,520000.0,35000.0,Diesel,Individual,Manual,First Owner,3738,13,Mid,Medium,0.0,Diesel_Manual,Hyundai,14.86,1
3738,CAR_003739,Maruti Alto LXi,2007,4461000.0,85000.0,Petrol,Individual,Manual,Second Owner,3739,18,Low,High,0.0,Petrol_Manual,Maruti,1.12,0
3739,CAR_003740,Toyota Innova 2.5 V Diesel 7-seater,2011,725000.0,110000.0,Diesel,Individual,Manual,Third Owner,3740,14,High,High,0.0,Diesel_Manual,Toyota,6.59,1
3740,CAR_003741,Maruti 800 AC,2006,55000.0,70000.0,Petrol,Individual,Manual,Second Owner,3741,19,Low,Medium,0.0,Petrol_Manual,Maruti,0.79,0
3741,CAR_003742,Honda City i DTEC VX,2014,600000.0,100000.0,LPG,Individual,Manual,First Owner,3742,11,Mid,High,0.0,Diesel_Manual,Honda,6.0,1
3742,CAR_003743,Volkswagen Vento New Diesel Highline,2013,330000.0,120000.0,Diesel,Individual,Manual,Second Owner,3743,12,Mid,High,0.0,Diesel_Manual,Volkswagen,2.75,1
3743,CAR_003744,Maruti Swift Dzire VDI,2015,470000.0,89000.0,Diesel,Individual,Manual,First Owner,3744,10,Mid,High,0.0,Diesel_Manual,Maruti,5.28,1
3744,CAR_003745,Maruti Swift ZDi,2013,365000.0,120000.0,Diesel,Individual,Manual,First Owner,3745,12,Mid,High,0.0,Diesel_Manual,Maruti,3.04,1
3745,CAR_003746,Hyundai EON Magna Plus,2013,250000.0,60000.0,Petrol,Individual,Manual,Second Owner,3746,12,Low,Medium,0.0,Petrol_Manual,Hyundai,6.25,0
3746,CAR_003747,Maruti Wagon R VXI BS IV,2017,430000.0,17000.0,Petrol,Individual,Manual,First Owner,3747,8,Mid,High,0.0,Petrol_Manual,Maruti,25.29,0
3747,CAR_003748,Maruti Swift VDI,2013,400000.0,125000.0,Diesel,Individual,Manual,Second Owner,3748,12,Mid,High,0.0,Diesel_Manual,Maruti,3.2,1
3748,CAR_003749,Maruti Swift Dzire ZXI,2015,615000.0,110000.0,Electric,Individual,Manual,Second Owner,3749,10,High,High,0.0,Petrol_Manual,Maruti,5.59,0
3749,CAR_003750,Ford Figo Diesel Titanium,2011,150000.0,120000.0,Diesel,Individual,Manual,Second Owner,3750,14,Low,High,0.0,Diesel_Manual,Ford,1.25,1
3750,CAR_003751,Ford Figo Diesel ZXI,2014,240000.0,90000.0,Diesel,Individual,Manual,First Owner,3751,11,Low,High,0.0,Diesel_Manual,Ford,2.67,1
3751,CAR_003752,Mahindra Scorpio S10 7 Seater,2014,800000.0,90000.0,Diesel,Individual,Manual,First Owner,3752,11,High,High,0.0,Diesel_Manual,Mahindra,8.89,1
3752,CAR_003753,Hyundai EON LPG Magna Plus,2015,240000.0,110000.0,LPG,Individual,Manual,First Owner,3753,10,Low,High,0.0,LPG_Manual,Hyundai,2.18,0
3753,CAR_003754,Ford Ecosport 1.5 DV5 MT Titanium,2014,600000.0,87000.0,Diesel,Individual,Manual,Second Owner,3754,11,Mid,High,0.0,Diesel_Manual,Ford,6.9,1
3754,CAR_003755,Mahindra Scorpio LX,2014,800000.0,60000.0,Diesel,Individual,Manual,First Owner,3755,11,High,Low,0.0,Diesel_Manual,Mahindra,53.33,1
3755,CAR_003756,Hyundai EON 1.0 Kappa Magna Plus,2015,160000.0,70000.0,Petrol,Individual,Manual,First Owner,3756,10,Low,Medium,0.0,Petrol_Manual,Hyundai,2.29,0
3756,CAR_003757,Ford Fiesta Classic 1.6 Duratec CLXI,2012,350000.0,38000.0,Petrol,Individual,Manual,First Owner,3757,13,Mid,Medium,0.0,Petrol_Manual,Ford,9.21,0
3757,CAR_003758,Honda City 1.5 V MT,2011,300000.0,64000.0,Petrol,Individual,Manual,Second Owner,3758,14,Low,Medium,0.0,Petrol_Manual,Honda,4.69,0
3758,CAR_003759,Tata Sumo Gold EX BSIII,2015,4461000.0,55250.0,Diesel,Individual,Manual,First Owner,3759,10,Mid,Medium,0.0,Diesel_Manual,Tata,7.22,1
3759,CAR_003760,Mahindra Bolero DI,2011,300000.0,110000.0,Diesel,Individual,Manual,First Owner,3760,14,Low,High,0.0,Diesel_Manual,Mahindra,2.73,1
3760,CAR_003761,Maruti A-Star AT VXI,2012,150000.0,70000.0,Petrol,Individual,Automatic,First Owner,3761,13,Low,Medium,0.0,Petrol_Automatic,Maruti,2.14,0
3761,CAR_003762,Chevrolet Cruze LTZ,2010,320000.0,60000.0,Diesel,Individual,Manual,Third Owner,3762,15,Mid,Medium,0.0,Diesel_Manual,Chevrolet,5.33,1
3762,CAR_003763,Tata Hexa XM,2018,1280000.0,17000.0,Diesel,Individual,Manual,First Owner,3763,7,Premium,Low,0.0,Diesel_Manual,Tata,75.29,1
3763,CAR_003764,Renault KWID 1.0 RXT Optional,2017,300000.0,13000.0,Petrol,Individual,Manual,Second Owner,3764,8,Low,Low,0.0,Petrol_Manual,Renault,23.08,0
3764,CAR_003765,Nissan Terrano XL Plus ICC WT20 SE,2016,1000000.0,60000.0,Diesel,Individual,Manual,First Owner,3765,9,High,High,0.0,Diesel_Manual,Nissan,16.67,1
3765,CAR_003766,Mitsubishi Pajero Sport 4X4,2012,1090000.0,120000.0,Diesel,Individual,Manual,Second Owner,3766,13,Premium,High,0.0,Diesel_Manual,Mitsubishi,9.08,1
3766,CAR_003767,Tata New Safari DICOR 2.2 GX 4x2 BS IV,2012,450000.0,97000.0,Petrol,Individual,Manual,Second Owner,3767,13,Mid,High,0.0,Diesel_Manual,Tata,4.64,1
3767,CAR_003768,Tata Indica GLS BS IV,2007,60000.0,60000.0,Diesel,Individual,Manual,Second Owner,3768,18,Low,High,0.0,Petrol_Manual,Tata,0.46,0
3768,CAR_003769,Maruti Swift Dzire ZXI,2013,500000.0,50000.0,Petrol,Individual,Manual,First Owner,3769,12,Mid,Medium,0.0,Petrol_Manual,Maruti,10.0,0
3769,CAR_003770,Tata Indigo CR4,2012,200000.0,120000.0,Diesel,Individual,Manual,Second Owner,3770,13,Low,High,0.0,Diesel_Manual,Tata,1.67,1
3770,CAR_003771,Ford Ecosport 1.5 DV5 MT Titanium,2015,650000.0,68000.0,Diesel,Individual,Manual,First Owner,3771,10,High,Medium,0.0,Diesel_Manual,Ford,9.56,1
3771,CAR_003772,Maruti Alto LXi,2006,60000.0,60000.0,Petrol,Individual,Manual,Second Owner,3772,19,Low,High,0.0,Petrol_Manual,Maruti,0.6,0
3772,CAR_003773,Maruti Ertiga VDI,2013,250999.0,80000.0,Diesel,Individual,Manual,First Owner,3773,12,Low,High,0.0,Diesel_Manual,Maruti,3.14,1
3773,CAR_003774,Hyundai Accent Executive CNG,2012,185000.0,67000.0,CNG,Dealer,Manual,Second Owner,3774,13,Low,Medium,0.0,CNG_Manual,Hyundai,2.76,0
3774,CAR_003775,Maruti Alto K10 VXI,2015,155000.0,60000.0,Petrol,Individual,Manual,First Owner,3775,10,Low,Medium,0.0,Petrol_Manual,Maruti,2.58,0
3775,CAR_003776,Maruti Swift VXI,2019,600000.0,5000.0,Petrol,Individual,Manual,First Owner,3776,6,Mid,Low,0.0,Petrol_Manual,Maruti,120.0,0
3776,CAR_003777,Renault KWID RXE,2016,300000.0,20000.0,Petrol,Individual,Manual,First Owner,3777,9,Low,Low,0.0,Petrol_Manual,Renault,15.0,0
3777,CAR_003778,Hyundai Creta 1.6 E Plus,2018,850000.0,12500.0,Petrol,Individual,Manual,First Owner,3778,7,High,Low,0.0,Petrol_Manual,Hyundai,68.0,0
3778,CAR_003779,Maruti Eeco CNG 5 Seater AC BSIV,2019,470000.0,4000.0,CNG,Individual,Manual,First Owner,3779,6,Mid,Low,0.0,CNG_Manual,Maruti,117.5,0
3779,CAR_003780,Ford Figo 1.2P Titanium MT,2019,600000.0,25000.0,Petrol,Individual,Manual,First Owner,3780,6,Mid,Low,0.0,Petrol_Manual,Ford,24.0,0
3780,CAR_003781,Toyota Innova 2.5 G (Diesel) 7 Seater,2013,4461000.0,70000.0,Diesel,Individual,Manual,Second Owner,3781,12,High,High,0.0,Diesel_Manual,Toyota,10.0,1
3781,CAR_003782,Maruti Wagon R LXI DUO BSIII,2007,70000.0,162000.0,LPG,Individual,Manual,First Owner,3782,18,Low,Very High,0.0,LPG_Manual,Maruti,0.43,0
3782,CAR_003783,Toyota Fortuner 3.0 Diesel,2010,1250000.0,205000.0,Diesel,Individual,Manual,Second Owner,3783,15,Premium,Very High,0.0,Diesel_Manual,Toyota,6.1,1
3783,CAR_003784,Audi A6 2.8 FSI,2008,650000.0,70000.0,Petrol,Individual,Automatic,Third Owner,3784,17,High,Medium,0.0,Petrol_Automatic,Audi,9.29,0
3784,CAR_003785,Hyundai Xcent 1.2 CRDi S,2018,450000.0,60000.0,Diesel,Individual,Manual,Second Owner,3785,7,Mid,Medium,0.0,Diesel_Manual,Hyundai,6.43,1
3785,CAR_003786,Mahindra XUV500 W8 2WD,2014,750000.0,100000.0,Diesel,Individual,Manual,Second Owner,3786,11,High,High,0.0,Diesel_Manual,Mahindra,7.5,1
3786,CAR_003787,Toyota Innova 2.5 G (Diesel) 7 Seater,2013,775000.0,70000.0,LPG,Individual,Manual,Second Owner,3787,12,High,Medium,0.0,Diesel_Manual,Toyota,11.07,1
3787,CAR_003788,Hyundai Santa Fe 4X4,2011,4461000.0,220000.0,Diesel,Individual,Manual,First Owner,3788,14,Low,Very High,0.0,Diesel_Manual,Hyundai,3.64,1
3788,CAR_003789,Honda Amaze E i-Vtech,2015,4461000.0,80000.0,Petrol,Individual,Manual,Second Owner,3789,10,Mid,High,0.0,Petrol_Manual,Honda,4.5,0
3789,CAR_003790,Toyota Corolla Altis Diesel D4DG,2010,250000.0,120000.0,Diesel,Individual,Manual,First Owner,3790,15,Low,High,0.0,Diesel_Manual,Toyota,2.08,1
3790,CAR_003791,Jeep Compass 2.0 Longitude Option BSIV,2017,1490000.0,22038.0,Diesel,Individual,Manual,First Owner,3791,8,Premium,Low,0.0,Diesel_Manual,Jeep,67.61,1
3791,CAR_003792,Nissan Terrano XL 110 PS,2013,600000.0,80000.0,Diesel,Individual,Manual,First Owner,3792,12,Mid,High,0.0,Diesel_Manual,Nissan,7.5,1
3792,CAR_003793,Maruti Alto LX,2010,120000.0,90000.0,Petrol,Individual,Manual,First Owner,3793,15,Low,High,0.0,Petrol_Manual,Maruti,1.33,0
3793,CAR_003794,Tata New Safari DICOR 2.2 EX 4x2,2012,320000.0,140000.0,Diesel,Individual,Manual,Second Owner,3794,13,Mid,High,0.0,Diesel_Manual,Tata,2.29,1
3794,CAR_003795,Mahindra Thar 4X2,2011,400000.0,120000.0,Diesel,Individual,Manual,Second Owner,3795,14,Mid,High,0.0,Diesel_Manual,Mahindra,3.33,1
3795,CAR_003796,Maruti Swift Dzire ZDI,2015,302000.0,60000.0,Diesel,Individual,Manual,First Owner,3796,10,Mid,Medium,0.0,Diesel_Manual,Maruti,5.03,1
3796,CAR_003797,Chevrolet Beat Diesel LT,2011,130000.0,60000.0,Diesel,Individual,Manual,Second Owner,3797,14,Low,High,0.0,Diesel_Manual,Chevrolet,1.62,1
3797,CAR_003798,Maruti Swift Dzire LXI Option,2014,150000.0,60000.0,Petrol,Individual,Manual,Second Owner,3798,11,Low,Medium,0.0,Petrol_Manual,Maruti,2.5,0
3798,CAR_003799,Tata Indica Vista Aura Plus 1.3 Quadrajet,2012,200000.0,90000.0,Diesel,Individual,Manual,Second Owner,3799,13,Low,High,0.0,Diesel_Manual,Tata,2.22,1
3799,CAR_003800,Toyota Innova 2.5 G1 BSIV,2012,950000.0,80000.0,Diesel,Individual,Manual,First Owner,3800,13,High,High,0.0,Diesel_Manual,Toyota,11.88,1
3800,CAR_003801,Maruti Swift Vdi BSIII,2008,120000.0,175000.0,Diesel,Individual,Manual,Second Owner,3801,17,Low,High,0.0,Diesel_Manual,Maruti,0.69,1
3801,CAR_003802,Mahindra XUV500 W7 BSIV,2019,1400000.0,10000.0,Diesel,Individual,Manual,First Owner,3802,6,Premium,High,0.0,Diesel_Manual,Mahindra,140.0,1
3802,CAR_003803,Hyundai i20 1.2 Asta Option,2017,620000.0,65000.0,Petrol,Individual,Manual,First Owner,3803,8,High,Medium,0.0,Petrol_Manual,Hyundai,9.54,0
3803,CAR_003804,Toyota Innova 2.5 GX 8 STR BSIV,2012,449000.0,90000.0,Diesel,Individual,Manual,First Owner,3804,13,Mid,High,0.0,Diesel_Manual,Toyota,4.99,1
3804,CAR_003805,Maruti Alto LXi,2007,100000.0,70000.0,Petrol,Individual,Manual,First Owner,3805,18,Low,Medium,0.0,Petrol_Manual,Maruti,1.43,0
3805,CAR_003806,Hyundai Grand i10 1.2 Kappa Sportz BSIV,2018,600000.0,2500.0,Petrol,Individual,Manual,First Owner,3806,7,Mid,Low,0.0,Petrol_Manual,Hyundai,240.0,0
3806,CAR_003807,Honda Amaze VX i-DTEC,2016,700000.0,35000.0,Diesel,Individual,Manual,First Owner,3807,9,High,High,0.0,Diesel_Manual,Honda,20.0,1
3807,CAR_003808,Chevrolet Beat Diesel LT,2012,250000.0,40000.0,Diesel,Individual,Manual,First Owner,3808,13,Low,Medium,0.0,Diesel_Manual,Chevrolet,6.25,1
3808,CAR_003809,Maruti Baleno RS 1.0 Petrol,2018,550000.0,63000.0,Petrol,Individual,Manual,First Owner,3809,7,Low,Medium,0.0,Petrol_Manual,Maruti,8.73,0
3809,CAR_003810,Skoda Superb Ambition 2.0 TDI CR AT,2013,4461000.0,88000.0,Diesel,Dealer,Automatic,First Owner,3810,12,Low,High,0.0,Diesel_Automatic,Skoda,7.67,1
3810,CAR_003811,Nissan Terrano XL Plus 85 PS,2015,4461000.0,55000.0,Diesel,Dealer,Manual,First Owner,3811,10,Low,Medium,0.0,Diesel_Manual,Nissan,9.64,1
3811,CAR_003812,Chevrolet Enjoy TCDi LT 7 Seater,2014,325000.0,71014.0,Diesel,Dealer,Manual,First Owner,3812,11,Mid,High,0.0,Diesel_Manual,Chevrolet,4.58,1
3812,CAR_003813,Renault Scala Diesel RxL,2015,370000.0,60000.0,Diesel,Dealer,Manual,First Owner,3813,10,Low,Medium,0.0,Diesel_Manual,Renault,6.17,1
3813,CAR_003814,Skoda Rapid 1.5 TDI Ambition,2014,360000.0,70000.0,Diesel,Individual,Manual,First Owner,3814,11,Mid,Medium,0.0,Diesel_Manual,Skoda,5.14,1
3814,CAR_003815,Skoda Superb LK 1.8 TSI AT,2012,650000.0,60000.0,Petrol,Dealer,Automatic,First Owner,3815,13,High,High,0.0,Petrol_Automatic,Skoda,10.83,0
3815,CAR_003816,Ford Fiesta 1.5 TDCi Titanium,2012,199000.0,92000.0,Diesel,Dealer,Manual,Third Owner,3816,13,Low,High,0.0,Diesel_Manual,Ford,2.16,1
3816,CAR_003817,Mahindra Quanto C6,2014,285000.0,89126.0,Electric,Individual,Manual,First Owner,3817,11,Low,High,0.0,Diesel_Manual,Mahindra,3.2,1
3817,CAR_003818,Volkswagen Jetta 1.9 TDI Comfortline DSG,2009,260000.0,134000.0,Diesel,Individual,Automatic,First Owner,3818,16,Low,High,0.0,Diesel_Automatic,Volkswagen,1.94,1
3818,CAR_003819,Maruti Ciaz 1.4 AT Zeta,2019,4461000.0,15000.0,Petrol,Dealer,Automatic,First Owner,3819,6,High,Low,0.0,Petrol_Automatic,Maruti,58.0,0
3819,CAR_003820,Hyundai Grand i10 1.2 CRDi Magna,2016,425000.0,60000.0,Diesel,Dealer,Manual,First Owner,3820,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.0,1
3820,CAR_003821,Chevrolet Cruze LTZ AT,2012,490000.0,42000.0,Diesel,Dealer,Automatic,First Owner,3821,13,Mid,High,0.0,Diesel_Automatic,Chevrolet,11.67,1
3821,CAR_003822,Mahindra XUV500 AT W8 FWD,2015,4461000.0,70000.0,Diesel,Individual,Automatic,Second Owner,3822,10,High,High,0.0,Diesel_Automatic,Mahindra,10.71,1
3822,CAR_003823,Honda City 1.5 EXI,2004,125000.0,110000.0,Petrol,Individual,Manual,Third Owner,3823,21,Low,High,0.0,Petrol_Manual,Honda,1.14,0
3823,CAR_003824,Fiat Linea 1.3 Emotion,2014,221000.0,131365.0,Diesel,Individual,Manual,First Owner,3824,11,Low,High,0.0,Diesel_Manual,Fiat,1.68,1
3824,CAR_003825,Volkswagen Vento 1.5 TDI Highline AT,2016,724000.0,48980.0,Diesel,Dealer,Automatic,First Owner,3825,9,High,Medium,0.0,Diesel_Automatic,Volkswagen,14.78,1
3825,CAR_003826,Honda Amaze VX AT i-Vtech,2014,440000.0,46000.0,Petrol,Dealer,Automatic,First Owner,3826,11,Mid,Medium,0.0,Petrol_Automatic,Honda,9.57,0
3826,CAR_003827,Chevrolet Spark 1.0 LS,2009,70000.0,60000.0,Petrol,Individual,Manual,Third Owner,3827,16,Low,Medium,0.0,Petrol_Manual,Chevrolet,1.17,0
3827,CAR_003828,Maruti Baleno Delta Diesel,2015,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,3828,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.81,1
3828,CAR_003829,Maruti S-Cross Alpha DDiS 200 SH,2016,790000.0,22000.0,Diesel,Dealer,Manual,First Owner,3829,9,High,Low,0.0,Diesel_Manual,Maruti,35.91,1
3829,CAR_003830,Hyundai i20 1.2 Asta Option,2017,4461000.0,37000.0,Petrol,Individual,Manual,First Owner,3830,8,High,Medium,0.0,Petrol_Manual,Hyundai,17.3,0
3830,CAR_003831,Hyundai Santro Xing GLS,2009,120000.0,70000.0,Petrol,Individual,Manual,Fourth & Above Owner,3831,16,Low,Medium,0.0,Petrol_Manual,Hyundai,1.71,0
3831,CAR_003832,Chevrolet Beat Diesel LT,2012,126000.0,98900.0,Diesel,Individual,Manual,Third Owner,3832,13,Low,High,0.0,Diesel_Manual,Chevrolet,1.27,1
3832,CAR_003833,Chevrolet Beat Diesel LT,2012,120000.0,125000.0,Diesel,Individual,Manual,First Owner,3833,13,Low,High,0.0,Diesel_Manual,Chevrolet,0.96,1
3833,CAR_003834,Mahindra TUV 300 T4 Plus,2017,650000.0,60000.0,Diesel,Individual,Manual,First Owner,3834,8,High,Medium,0.0,Diesel_Manual,Mahindra,10.83,1
3834,CAR_003835,Hyundai Grand i10 Asta,2014,450000.0,80000.0,Petrol,Individual,Manual,Second Owner,3835,11,Mid,High,0.0,Petrol_Manual,Hyundai,5.62,0
3835,CAR_003836,Maruti Zen Estilo LXI Green (CNG),2010,110000.0,80000.0,CNG,Individual,Manual,Fourth & Above Owner,3836,15,Low,High,0.0,CNG_Manual,Maruti,1.38,0
3836,CAR_003837,Mahindra Bolero Power Plus Plus AC BSIV PS,2017,4461000.0,40000.0,Diesel,Individual,Manual,First Owner,3837,8,Mid,Medium,0.0,Diesel_Manual,Mahindra,15.0,1
3837,CAR_003838,Mahindra XUV500 W8 4WD,2012,620000.0,60000.0,Petrol,Individual,Manual,Second Owner,3838,13,High,High,0.0,Diesel_Manual,Mahindra,7.75,1
3838,CAR_003839,Ford Ecosport 1.5 DV5 MT Titanium,2014,600000.0,30000.0,Diesel,Individual,Manual,First Owner,3839,11,Mid,Low,0.0,Diesel_Manual,Ford,20.0,1
3839,CAR_003840,Mahindra Bolero Power Plus ZLX,2018,795000.0,35000.0,Diesel,Individual,Manual,First Owner,3840,7,High,Medium,0.0,Diesel_Manual,Mahindra,22.71,1
3840,CAR_003841,Tata Winger Deluxe - Hi Roof (AC),2011,300000.0,180000.0,Diesel,Individual,Manual,Third Owner,3841,14,Low,Very High,0.0,Diesel_Manual,Tata,1.67,1
3841,CAR_003842,Renault Lodgy 85PS RxL,2015,495000.0,60000.0,Diesel,Individual,Manual,First Owner,3842,10,Mid,Medium,0.0,Diesel_Manual,Renault,8.25,1
3842,CAR_003843,Audi A4 3.0 TDI Quattro,2013,1580000.0,86000.0,Diesel,Dealer,Automatic,First Owner,3843,12,Premium,High,0.0,Diesel_Automatic,Audi,18.37,1
3843,CAR_003844,Audi Q3 2.0 TDI Quattro Premium Plus,2015,1750000.0,127643.0,Diesel,Dealer,Automatic,First Owner,3844,10,Low,High,0.0,Diesel_Automatic,Audi,13.71,1
3844,CAR_003845,Tata Hexa XTA,2018,1350000.0,60000.0,Diesel,Individual,Automatic,First Owner,3845,7,Low,Medium,0.0,Diesel_Automatic,Tata,39.71,1
3845,CAR_003846,Mahindra Xylo D4,2014,4461000.0,145000.0,Diesel,Individual,Manual,First Owner,3846,11,Mid,High,0.0,Diesel_Manual,Mahindra,2.41,1
3846,CAR_003847,Datsun RediGO AMT 1.0 S,2018,4461000.0,25000.0,Petrol,Individual,Automatic,First Owner,3847,7,Low,Low,0.0,Petrol_Automatic,Datsun,12.0,0
3847,CAR_003848,Tata Zest Revotron 1.2T XMS,2018,400000.0,54000.0,Petrol,Dealer,Manual,Second Owner,3848,7,Mid,Medium,0.0,Petrol_Manual,Tata,7.41,0
3848,CAR_003849,Audi A6 2.0 TDI  Design Edition,2014,1750000.0,102354.0,Diesel,Dealer,Automatic,First Owner,3849,11,Premium,High,0.0,Diesel_Automatic,Audi,17.1,1
3849,CAR_003850,Maruti Zen LX,1999,70000.0,70000.0,Petrol,Individual,Manual,Fourth & Above Owner,3850,26,Low,Medium,0.0,Petrol_Manual,Maruti,1.0,0
3850,CAR_003851,Nissan Sunny XL D,2018,500000.0,70000.0,Diesel,Individual,Manual,First Owner,3851,7,Mid,Medium,0.0,Diesel_Manual,Nissan,7.14,1
3851,CAR_003852,Audi Q3 35 TDI Quattro Technology,2017,2575000.0,36000.0,Diesel,Dealer,Manual,First Owner,3852,8,Premium,Medium,0.0,Diesel_Automatic,Audi,71.53,1
3852,CAR_003853,Honda Brio VX,2014,345000.0,58000.0,Petrol,Dealer,Manual,First Owner,3853,11,Mid,Medium,0.0,Petrol_Manual,Honda,5.95,0
3853,CAR_003854,Tata Indigo CR4,2013,220000.0,100000.0,Diesel,Individual,Manual,First Owner,3854,12,Low,High,0.0,Diesel_Manual,Tata,2.2,1
3854,CAR_003855,Audi Q5 2.0 TFSI Quattro Premium Plus,2014,1850000.0,62237.0,Petrol,Dealer,Automatic,First Owner,3855,11,Premium,Medium,0.0,Petrol_Automatic,Audi,29.73,0
3855,CAR_003856,Hyundai Santro Xing XL eRLX Euro III,2005,114999.0,90000.0,Petrol,Dealer,Manual,Second Owner,3856,20,Low,High,0.0,Petrol_Manual,Hyundai,1.28,0
3856,CAR_003857,Maruti 800 AC,1999,50000.0,100000.0,Petrol,Individual,Manual,Second Owner,3857,26,Low,High,0.0,Petrol_Manual,Maruti,0.5,0
3857,CAR_003858,Honda Jazz Select Edition,2017,585000.0,14000.0,Petrol,Dealer,Manual,First Owner,3858,8,Mid,Low,0.0,Petrol_Manual,Honda,41.79,0
3858,CAR_003859,Ford EcoSport 1.5 TDCi Titanium BSIV,2018,950000.0,21394.0,Diesel,Dealer,Manual,First Owner,3859,7,Low,Low,0.0,Diesel_Manual,Ford,44.4,1
3859,CAR_003860,Maruti Swift Dzire AMT VXI,2018,625000.0,13800.0,Petrol,Individual,Automatic,First Owner,3860,7,High,Low,0.0,Petrol_Automatic,Maruti,45.29,0
3860,CAR_003861,Hyundai Creta 1.6 SX Automatic,2017,1035000.0,23000.0,Petrol,Dealer,Automatic,First Owner,3861,8,Premium,Low,0.0,Petrol_Automatic,Hyundai,45.0,0
3861,CAR_003862,Datsun GO T Petrol,2015,310000.0,32686.0,Petrol,Dealer,Manual,First Owner,3862,10,Mid,High,0.0,Petrol_Manual,Datsun,9.48,0
3862,CAR_003863,Mahindra Xylo D4,2017,600000.0,99700.0,Diesel,Individual,Manual,First Owner,3863,8,Mid,High,0.0,Diesel_Manual,Mahindra,6.02,1
3863,CAR_003864,Toyota Fortuner 2.8 4WD AT BSIV,2019,3100000.0,5000.0,Diesel,Individual,Automatic,First Owner,3864,6,Premium,Low,0.0,Diesel_Automatic,Toyota,620.0,1
3864,CAR_003865,Honda City i-DTEC SV,2014,650000.0,49654.0,Diesel,Dealer,Manual,First Owner,3865,11,High,Medium,0.0,Diesel_Manual,Honda,13.09,1
3865,CAR_003866,Hyundai Grand i10 Asta,2016,550000.0,60000.0,Petrol,Dealer,Manual,First Owner,3866,9,Mid,High,0.0,Petrol_Manual,Hyundai,12.1,0
3866,CAR_003867,Hyundai Creta 1.6 CRDi SX,2017,1260000.0,39221.0,Diesel,Dealer,Manual,First Owner,3867,8,Premium,Medium,0.0,Diesel_Manual,Hyundai,32.13,1
3867,CAR_003868,Hyundai i20 Sportz 1.2,2016,660000.0,60000.0,Petrol,Dealer,Manual,First Owner,3868,9,High,Medium,0.0,Petrol_Manual,Hyundai,13.69,0
3868,CAR_003869,Hyundai i20 Sportz 1.2,2017,700000.0,11114.0,Petrol,Dealer,Manual,First Owner,3869,8,High,Low,0.0,Petrol_Manual,Hyundai,62.98,0
3869,CAR_003870,Hyundai Getz 1.5 CRDi GVS,2008,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,3870,17,Low,Medium,0.0,Diesel_Manual,Hyundai,3.9,1
3870,CAR_003871,Volkswagen Vento 1.5 TDI Highline,2013,475000.0,80000.0,Diesel,Dealer,Manual,First Owner,3871,12,Low,High,0.0,Diesel_Manual,Volkswagen,5.94,1
3871,CAR_003872,Mercedes-Benz E-Class 280 CDI Elegance,2008,795000.0,98600.0,Diesel,Dealer,Automatic,First Owner,3872,17,High,High,0.0,Diesel_Automatic,Mercedes-Benz,8.06,1
3872,CAR_003873,Audi RS7 2015-2019 Sportback Performance,2016,8900000.0,13000.0,Petrol,Dealer,Automatic,First Owner,3873,9,Low,Low,0.0,Petrol_Automatic,Audi,684.62,0
3873,CAR_003874,Audi Q5 3.0 TDI Quattro Technology,2015,3200000.0,30000.0,Diesel,Dealer,Automatic,First Owner,3874,10,Premium,Low,0.0,Diesel_Automatic,Audi,106.67,1
3874,CAR_003875,MG Hector Smart AT,2019,1860000.0,18000.0,CNG,Dealer,Automatic,First Owner,3875,6,Premium,High,0.0,Petrol_Automatic,MG,103.33,0
3875,CAR_003876,Land Rover Range Rover 4.4 Diesel LWB Vogue SE,2010,4200000.0,100000.0,Diesel,Dealer,Automatic,First Owner,3876,15,Premium,High,0.0,Diesel_Automatic,Land,42.0,1
3876,CAR_003877,Honda City VX CVT,2015,850000.0,58000.0,LPG,Dealer,Automatic,First Owner,3877,10,High,High,0.0,Petrol_Automatic,Honda,14.66,0
3877,CAR_003878,Mahindra Xylo D4,2017,585000.0,85441.0,Diesel,Dealer,Manual,First Owner,3878,8,Mid,High,0.0,Diesel_Manual,Mahindra,6.85,1
3878,CAR_003879,Mahindra Verito 1.5 D4 BSIV,2015,380000.0,64541.0,Diesel,Dealer,Manual,First Owner,3879,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.89,1
3879,CAR_003880,Maruti Swift Dzire VDI,2019,650000.0,35000.0,Diesel,Individual,Manual,First Owner,3880,6,High,High,0.0,Diesel_Manual,Maruti,18.57,1
3880,CAR_003881,Maruti Swift ZXI Plus,2018,600000.0,30000.0,Petrol,Individual,Manual,First Owner,3881,7,Mid,Low,0.0,Petrol_Manual,Maruti,20.0,0
3881,CAR_003882,Nissan Micra Active XV,2017,4461000.0,16267.0,Electric,Dealer,Manual,First Owner,3882,8,Mid,Low,0.0,Petrol_Manual,Nissan,26.74,0
3882,CAR_003883,Volkswagen Vento 1.5 TDI Highline,2013,525000.0,71500.0,Diesel,Dealer,Manual,First Owner,3883,12,Mid,High,0.0,Diesel_Manual,Volkswagen,7.34,1
3883,CAR_003884,BMW 5 Series 520d Luxury Line,2019,4461000.0,60000.0,Diesel,Dealer,Automatic,First Owner,3884,6,Premium,Low,0.0,Diesel_Automatic,BMW,369.26,1
3884,CAR_003885,Chevrolet Aveo U-VA 1.2 LS,2008,160000.0,80000.0,Petrol,Individual,Manual,First Owner,3885,17,Low,High,0.0,Petrol_Manual,Chevrolet,2.0,0
3885,CAR_003886,Maruti Zen Estilo 1.1 LXI BSIII,2007,160000.0,90000.0,Petrol,Individual,Manual,Second Owner,3886,18,Low,High,0.0,Petrol_Manual,Maruti,1.78,0
3886,CAR_003887,Chevrolet Spark 1.0 LT,2012,125000.0,28000.0,Petrol,Individual,Manual,First Owner,3887,13,Low,Low,0.0,Petrol_Manual,Chevrolet,4.46,0
3887,CAR_003888,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,430000.0,200000.0,Diesel,Individual,Manual,First Owner,3888,10,Low,Very High,0.0,Diesel_Manual,Mahindra,2.15,1
3888,CAR_003889,Hyundai Creta 1.6 VTVT S,2017,900000.0,14700.0,Petrol,Individual,Manual,First Owner,3889,8,High,Low,0.0,Petrol_Manual,Hyundai,61.22,0
3889,CAR_003890,Maruti Alto LX,2000,61000.0,90000.0,Petrol,Individual,Manual,Second Owner,3890,25,Low,High,0.0,Petrol_Manual,Maruti,0.68,0
3890,CAR_003891,Maruti Wagon R LXI BS IV,2017,400000.0,50000.0,Petrol,Individual,Manual,First Owner,3891,8,Low,Medium,0.0,Petrol_Manual,Maruti,8.0,0
3891,CAR_003892,Hyundai Santro Xing XS,2009,150000.0,100000.0,Petrol,Individual,Manual,Second Owner,3892,16,Low,High,0.0,Petrol_Manual,Hyundai,1.5,0
3892,CAR_003893,Volkswagen Polo GT 1.0 TSI,2019,950000.0,24000.0,Petrol,Individual,Automatic,First Owner,3893,6,Low,Low,0.0,Petrol_Automatic,Volkswagen,39.58,0
3893,CAR_003894,Tata Indigo CR4,2013,220000.0,100000.0,Diesel,Individual,Manual,First Owner,3894,12,Low,High,0.0,Diesel_Manual,Tata,2.2,1
3894,CAR_003895,Hyundai Getz GLX,2007,110000.0,90000.0,Diesel,Individual,Manual,First Owner,3895,18,Low,High,0.0,Petrol_Manual,Hyundai,1.22,0
3895,CAR_003896,Tata Bolt Revotron XM,2015,375000.0,60000.0,Petrol,Individual,Manual,First Owner,3896,10,Mid,Low,0.0,Petrol_Manual,Tata,12.5,0
3896,CAR_003897,Datsun GO Plus T Option,2017,350000.0,80000.0,Petrol,Individual,Manual,First Owner,3897,8,Mid,High,0.0,Petrol_Manual,Datsun,4.38,0
3897,CAR_003898,Hyundai i20 Magna 1.4 CRDi (Diesel),2012,310000.0,92686.0,Diesel,Dealer,Manual,Second Owner,3898,13,Mid,High,0.0,Diesel_Manual,Hyundai,3.34,1
3898,CAR_003899,Tata Indica GLS BS IV,2010,90000.0,300000.0,Petrol,Individual,Manual,Third Owner,3899,15,Low,High,0.0,Petrol_Manual,Tata,0.3,0
3899,CAR_003900,Volkswagen Vento Petrol Highline,2011,350000.0,90000.0,Petrol,Individual,Manual,Second Owner,3900,14,Mid,High,0.0,Petrol_Manual,Volkswagen,3.89,0
3900,CAR_003901,Hyundai Santro LS zipPlus,2003,100000.0,60000.0,Petrol,Individual,Manual,First Owner,3901,22,Low,Medium,0.0,Petrol_Manual,Hyundai,1.67,0
3901,CAR_003902,Mahindra XUV500 AT W6 2WD,2017,1300000.0,40000.0,Diesel,Individual,Automatic,First Owner,3902,8,Premium,Medium,0.0,Diesel_Automatic,Mahindra,32.5,1
3902,CAR_003903,Maruti Wagon R LXI,2004,70000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,3903,21,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
3903,CAR_003904,Chevrolet Sail Hatchback LT ABS,2013,225000.0,60000.0,CNG,Individual,Manual,First Owner,3904,12,Low,High,0.0,Diesel_Manual,Chevrolet,2.81,1
3904,CAR_003905,Honda City 1.3 EXI,2002,145000.0,100000.0,Petrol,Individual,Manual,First Owner,3905,23,Low,High,0.0,Petrol_Manual,Honda,1.45,0
3905,CAR_003906,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2018,930000.0,60000.0,Diesel,Individual,Manual,First Owner,3906,7,High,Low,0.0,Diesel_Manual,Ford,46.5,1
3906,CAR_003907,Maruti Swift VXI,2019,550000.0,51000.0,Petrol,Individual,Manual,First Owner,3907,6,Mid,Medium,0.0,Petrol_Manual,Maruti,10.78,0
3907,CAR_003908,Maruti Alto LX BSIII,2008,85000.0,120000.0,Petrol,Individual,Manual,Second Owner,3908,17,Low,High,0.0,Petrol_Manual,Maruti,0.71,0
3908,CAR_003909,Hyundai EON Era Plus Sports Edition,2014,280000.0,55000.0,Petrol,Individual,Manual,First Owner,3909,11,Low,Medium,0.0,Petrol_Manual,Hyundai,5.09,0
3909,CAR_003910,Hyundai EON Magna Plus,2018,360000.0,6000.0,Petrol,Individual,Manual,First Owner,3910,7,Mid,Low,0.0,Petrol_Manual,Hyundai,60.0,0
3910,CAR_003911,Maruti Alto LXi,2011,150000.0,40000.0,Petrol,Individual,Manual,Second Owner,3911,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.75,0
3911,CAR_003912,Skoda Rapid 1.6 MPI AT Ambition BSIV,2017,900000.0,15000.0,Petrol,Individual,Automatic,First Owner,3912,8,High,Low,0.0,Petrol_Automatic,Skoda,60.0,0
3912,CAR_003913,Ambassador CLASSIC 1500 DSL AC,2005,120000.0,50000.0,Diesel,Individual,Manual,Second Owner,3913,20,Low,Medium,0.0,Diesel_Manual,Ambassador,2.4,1
3913,CAR_003914,Fiat Linea T Jet Emotion,2010,215000.0,90000.0,Petrol,Individual,Manual,Second Owner,3914,15,Low,High,0.0,Petrol_Manual,Fiat,2.39,0
3914,CAR_003915,Renault Scala Diesel RxL,2013,550000.0,38500.0,Diesel,Individual,Manual,First Owner,3915,12,Mid,Medium,0.0,Diesel_Manual,Renault,14.29,1
3915,CAR_003916,Tata Nexon 1.2 Revotron XZ Plus Dual Tone,2018,775000.0,35000.0,Petrol,Individual,Manual,First Owner,3916,7,High,High,0.0,Petrol_Manual,Tata,22.14,0
3916,CAR_003917,Maruti Gypsy King HT BSIV,2001,409999.0,49359.0,Petrol,Individual,Manual,Third Owner,3917,24,Mid,Medium,0.0,Petrol_Manual,Maruti,8.31,0
3917,CAR_003918,Maruti 800 AC Uniq,2005,4461000.0,70000.0,Petrol,Individual,Manual,Second Owner,3918,20,Low,Medium,0.0,Petrol_Manual,Maruti,0.93,0
3918,CAR_003919,Maruti Wagon R LXI Minor,2008,200000.0,90000.0,Petrol,Individual,Manual,Second Owner,3919,17,Low,High,0.0,Petrol_Manual,Maruti,2.22,0
3919,CAR_003920,Maruti Wagon R VXI BS IV,2010,240000.0,50000.0,Petrol,Individual,Manual,Second Owner,3920,15,Low,High,0.0,Petrol_Manual,Maruti,4.8,0
3920,CAR_003921,Toyota Innova 2.5 VX 8 STR,2012,800000.0,108731.0,Diesel,Individual,Manual,First Owner,3921,13,High,High,0.0,Diesel_Manual,Toyota,7.36,1
3921,CAR_003922,Toyota Innova 2.5 EV Diesel MS 7 Str BSIII,2010,400000.0,120000.0,LPG,Individual,Manual,First Owner,3922,15,Mid,High,0.0,Diesel_Manual,Toyota,3.33,1
3922,CAR_003923,Tata Sumo Gold EX BSIII,2016,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,3923,9,Mid,Medium,0.0,Diesel_Manual,Tata,8.33,1
3923,CAR_003924,Mahindra Renault Logan 1.5 DLX Diesel,2007,150000.0,90000.0,Diesel,Individual,Manual,Third Owner,3924,18,Low,High,0.0,Diesel_Manual,Mahindra,1.67,1
3924,CAR_003925,Toyota Etios GD,2014,500000.0,140000.0,Diesel,Individual,Manual,Second Owner,3925,11,Mid,High,0.0,Diesel_Manual,Toyota,3.57,1
3925,CAR_003926,Hyundai EON Magna Plus,2014,4461000.0,80000.0,Petrol,Individual,Manual,First Owner,3926,11,Mid,High,0.0,Petrol_Manual,Hyundai,4.0,0
3926,CAR_003927,Nissan Micra Diesel XV,2013,300000.0,90000.0,Diesel,Individual,Manual,First Owner,3927,12,Low,High,0.0,Diesel_Manual,Nissan,3.33,1
3927,CAR_003928,Renault Lodgy Stepway 85PS RXZ 8S,2017,650000.0,40000.0,Electric,Individual,Manual,First Owner,3928,8,High,Medium,0.0,Diesel_Manual,Renault,16.25,1
3928,CAR_003929,Maruti Alto 800 LXI,2018,285000.0,14000.0,Petrol,Individual,Manual,First Owner,3929,7,Low,High,0.0,Petrol_Manual,Maruti,20.36,0
3929,CAR_003930,Tata Aria Pleasure 4x2,2012,360000.0,70000.0,Diesel,Individual,Manual,Second Owner,3930,13,Mid,Medium,0.0,Diesel_Manual,Tata,5.14,1
3930,CAR_003931,Honda Accord 2.4 MT,2009,450000.0,55000.0,Petrol,Individual,Manual,Second Owner,3931,16,Low,Medium,0.0,Petrol_Manual,Honda,8.18,0
3931,CAR_003932,Maruti Alto LXi,2011,4461000.0,60000.0,Petrol,Individual,Manual,Third Owner,3932,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.92,0
3932,CAR_003933,Chevrolet Cruze LT,2011,350000.0,74000.0,Diesel,Individual,Manual,Second Owner,3933,14,Mid,High,0.0,Diesel_Manual,Chevrolet,4.73,1
3933,CAR_003934,Ford Figo Aspire 1.5 TDCi Titanium,2020,530000.0,45000.0,Diesel,Dealer,Manual,First Owner,3934,5,Mid,Medium,0.0,Diesel_Manual,Ford,11.78,1
3934,CAR_003935,Honda City i-DTEC ZX,2017,1025000.0,38000.0,Diesel,Dealer,Manual,First Owner,3935,8,Premium,Medium,0.0,Diesel_Manual,Honda,26.97,1
3935,CAR_003936,Honda BR-V i-DTEC VX MT,2017,950000.0,24000.0,Diesel,Dealer,Manual,First Owner,3936,8,High,Low,0.0,Diesel_Manual,Honda,39.58,1
3936,CAR_003937,Honda WR-V i-VTEC VX,2017,725000.0,29976.0,Petrol,Dealer,Manual,First Owner,3937,8,High,Low,0.0,Petrol_Manual,Honda,24.19,0
3937,CAR_003938,Honda City i VTEC VX,2017,919999.0,60000.0,Petrol,Dealer,Manual,First Owner,3938,8,High,Medium,0.0,Petrol_Manual,Honda,22.44,0
3938,CAR_003939,Honda Jazz 1.5 V i DTEC,2018,700000.0,15000.0,Diesel,Dealer,Manual,First Owner,3939,7,High,Low,0.0,Diesel_Manual,Honda,46.67,1
3939,CAR_003940,Honda Jazz 1.5 VX i DTEC,2017,725000.0,36000.0,Diesel,Dealer,Manual,First Owner,3940,8,High,Medium,0.0,Diesel_Manual,Honda,20.14,1
3940,CAR_003941,Honda BR-V i-VTEC VX MT,2017,840000.0,30646.0,CNG,Dealer,Manual,First Owner,3941,8,High,Medium,0.0,Petrol_Manual,Honda,27.41,0
3941,CAR_003942,Maruti Wagon R VXI BS IV,2015,365000.0,60000.0,Petrol,Dealer,Manual,First Owner,3942,10,Mid,Low,0.0,Petrol_Manual,Maruti,15.47,0
3942,CAR_003943,Honda City i DTEC SV,2014,665000.0,71318.0,Diesel,Dealer,Manual,First Owner,3943,11,Low,High,0.0,Diesel_Manual,Honda,9.32,1
3943,CAR_003944,Maruti 800 Std,2003,50000.0,60000.0,Petrol,Individual,Manual,First Owner,3944,22,Low,High,0.0,Petrol_Manual,Maruti,0.64,0
3944,CAR_003945,Hyundai EON Era Plus,2014,225000.0,60000.0,Petrol,Individual,Manual,First Owner,3945,11,Low,High,0.0,Petrol_Manual,Hyundai,7.5,0
3945,CAR_003946,Hyundai i10 Era,2012,4461000.0,45000.0,Petrol,Individual,Manual,Second Owner,3946,13,Low,High,0.0,Petrol_Manual,Hyundai,5.11,0
3946,CAR_003947,Hyundai Santro Xing GL,2008,145000.0,67000.0,Petrol,Individual,Manual,First Owner,3947,17,Low,Medium,0.0,Petrol_Manual,Hyundai,2.16,0
3947,CAR_003948,Ford Figo Aspire Facelift,2018,4461000.0,14681.0,Diesel,Dealer,Manual,First Owner,3948,7,High,Low,0.0,Diesel_Manual,Ford,42.5,1
3948,CAR_003949,Hyundai Verna 1.4 VTVT,2015,500000.0,25000.0,Petrol,Individual,Manual,First Owner,3949,10,Mid,Low,0.0,Petrol_Manual,Hyundai,20.0,0
3949,CAR_003950,Ford EcoSport 1.5 Ti VCT MT Titanium BSIV,2017,750000.0,18054.0,Petrol,Dealer,Manual,First Owner,3950,8,High,Low,0.0,Petrol_Manual,Ford,41.54,0
3950,CAR_003951,Honda City i DTEC VX,2015,785000.0,40000.0,Diesel,Individual,Manual,First Owner,3951,10,High,Medium,0.0,Diesel_Manual,Honda,19.62,1
3951,CAR_003952,Hyundai i10 Sportz,2011,225000.0,30000.0,Petrol,Individual,Manual,First Owner,3952,14,Low,High,0.0,Petrol_Manual,Hyundai,7.5,0
3952,CAR_003953,Hyundai Xcent 1.1 CRDi Base,2017,300000.0,80000.0,Diesel,Individual,Manual,First Owner,3953,8,Low,High,0.0,Diesel_Manual,Hyundai,3.75,1
3953,CAR_003954,Hyundai i20 Asta 1.4 CRDi,2016,680000.0,20000.0,Diesel,Individual,Manual,First Owner,3954,9,High,Low,0.0,Diesel_Manual,Hyundai,34.0,1
3954,CAR_003955,Mahindra TUV 300 T8,2016,4461000.0,40000.0,Diesel,Individual,Manual,First Owner,3955,9,High,Medium,0.0,Diesel_Manual,Mahindra,17.5,1
3955,CAR_003956,Tata Indigo CS eLS BS IV,2011,180000.0,120000.0,Diesel,Individual,Manual,First Owner,3956,14,Low,High,0.0,Diesel_Manual,Tata,1.5,1
3956,CAR_003957,Ford Fiesta 1.4 SXi TDCi ABS,2008,270000.0,120000.0,Diesel,Individual,Manual,Second Owner,3957,17,Low,High,0.0,Diesel_Manual,Ford,2.25,1
3957,CAR_003958,Maruti Swift Dzire VXi,2010,370000.0,60000.0,Petrol,Individual,Manual,Second Owner,3958,15,Mid,High,0.0,Petrol_Manual,Maruti,3.81,0
3958,CAR_003959,Mahindra Xylo H8 ABS with Airbags,2018,1100000.0,60000.0,Diesel,Individual,Manual,First Owner,3959,7,Premium,Medium,0.0,Diesel_Manual,Mahindra,15.71,1
3959,CAR_003960,Hyundai Accent CRDi,2004,120000.0,120000.0,Diesel,Individual,Manual,Second Owner,3960,21,Low,High,0.0,Diesel_Manual,Hyundai,1.0,1
3960,CAR_003961,Skoda Rapid 1.6 TDI PRESTIGE,2013,430000.0,60000.0,Diesel,Individual,Manual,Second Owner,3961,12,Mid,High,0.0,Diesel_Manual,Skoda,4.89,1
3961,CAR_003962,Maruti Swift ZDi,2014,550000.0,38406.0,Diesel,Dealer,Manual,First Owner,3962,11,Mid,Medium,0.0,Diesel_Manual,Maruti,14.32,1
3962,CAR_003963,Hyundai Santro Xing GLS,2014,4461000.0,54350.0,Petrol,Dealer,Manual,First Owner,3963,11,Mid,Medium,0.0,Petrol_Manual,Hyundai,5.89,0
3963,CAR_003964,Maruti Wagon R VXI BS IV,2015,390000.0,32260.0,Petrol,Dealer,Manual,First Owner,3964,10,Mid,High,0.0,Petrol_Manual,Maruti,12.09,0
3964,CAR_003965,Renault Duster 85PS Diesel RxE,2014,550000.0,58231.0,Diesel,Dealer,Manual,First Owner,3965,11,Mid,Medium,0.0,Diesel_Manual,Renault,9.45,1
3965,CAR_003966,Maruti Swift VXI,2012,370000.0,59858.0,Petrol,Dealer,Manual,First Owner,3966,13,Mid,Medium,0.0,Petrol_Manual,Maruti,6.18,0
3966,CAR_003967,Maruti Alto LXi BSIII,2010,4461000.0,73350.0,LPG,Dealer,Manual,First Owner,3967,15,Low,High,0.0,Petrol_Manual,Maruti,2.59,0
3967,CAR_003968,Renault Duster 110PS Diesel RxZ,2013,4461000.0,88473.0,Diesel,Dealer,Manual,First Owner,3968,12,Mid,High,0.0,Diesel_Manual,Renault,5.76,1
3968,CAR_003969,Honda City S,2013,450000.0,96987.0,Petrol,Dealer,Manual,First Owner,3969,12,Mid,High,0.0,Petrol_Manual,Honda,4.64,0
3969,CAR_003970,Mercedes-Benz GLS 2016-2020 350d 4MATIC,2016,5500000.0,77350.0,Diesel,Dealer,Automatic,First Owner,3970,9,Premium,High,0.0,Diesel_Automatic,Mercedes-Benz,71.11,1
3970,CAR_003971,Maruti Swift Dzire VDI,2015,550000.0,61187.0,Diesel,Dealer,Manual,First Owner,3971,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.99,1
3971,CAR_003972,Mahindra Xylo E6,2009,400000.0,68350.0,Diesel,Dealer,Manual,First Owner,3972,16,Mid,Medium,0.0,Diesel_Manual,Mahindra,5.85,1
3972,CAR_003973,Ford EcoSport 1.5 TDCi Titanium BSIV,2015,585000.0,81150.0,Diesel,Dealer,Manual,First Owner,3973,10,Mid,High,0.0,Diesel_Manual,Ford,7.21,1
3973,CAR_003974,Hyundai Grand i10 1.2 Kappa Asta,2019,4461000.0,30000.0,Petrol,Individual,Manual,First Owner,3974,6,Mid,Low,0.0,Petrol_Manual,Hyundai,16.67,0
3974,CAR_003975,Maruti Ciaz ZXi,2015,800000.0,30000.0,Petrol,Individual,Manual,First Owner,3975,10,High,Low,0.0,Petrol_Manual,Maruti,26.67,0
3975,CAR_003976,Maruti Swift Vdi BSIII,2009,265000.0,120000.0,Diesel,Individual,Manual,Second Owner,3976,16,Low,High,0.0,Diesel_Manual,Maruti,2.21,1
3976,CAR_003977,Ford Figo Diesel Titanium,2012,175000.0,60000.0,Diesel,Individual,Manual,First Owner,3977,13,Low,High,0.0,Diesel_Manual,Ford,0.97,1
3977,CAR_003978,Fiat Palio 1.2 Sport,2007,125000.0,60000.0,Petrol,Individual,Manual,Second Owner,3978,18,Low,Medium,0.0,Petrol_Manual,Fiat,2.5,0
3978,CAR_003979,Tata Manza Aqua Quadrajet BS IV,2010,4461000.0,80000.0,Diesel,Individual,Manual,First Owner,3979,15,Low,High,0.0,Diesel_Manual,Tata,2.06,1
3979,CAR_003980,Mahindra Verito 1.5 D2 BSIII,2011,150000.0,280000.0,Diesel,Individual,Manual,First Owner,3980,14,Low,High,0.0,Diesel_Manual,Mahindra,0.54,1
3980,CAR_003981,Hyundai Creta 1.6 CRDi SX Option,2015,1025000.0,90000.0,Diesel,Individual,Manual,Second Owner,3981,10,Premium,High,0.0,Diesel_Manual,Hyundai,11.39,1
3981,CAR_003982,Toyota Innova 2.5 VX (Diesel) 8 Seater,2014,1030000.0,250000.0,Diesel,Individual,Manual,Second Owner,3982,11,Premium,Very High,0.0,Diesel_Manual,Toyota,4.12,1
3982,CAR_003983,Chevrolet Cruze LT,2012,380000.0,90000.0,Electric,Individual,Manual,Fourth & Above Owner,3983,13,Mid,High,0.0,Diesel_Manual,Chevrolet,4.22,1
3983,CAR_003984,Maruti Swift Dzire LDi,2010,250000.0,110000.0,Diesel,Individual,Manual,Third Owner,3984,15,Low,High,0.0,Diesel_Manual,Maruti,2.27,1
3984,CAR_003985,Maruti Zen LX,2000,85000.0,80000.0,Petrol,Individual,Manual,Second Owner,3985,25,Low,High,0.0,Petrol_Manual,Maruti,1.06,0
3985,CAR_003986,Fiat Punto EVO 1.3 Dynamic,2016,500000.0,35000.0,Diesel,Individual,Manual,First Owner,3986,9,Mid,Medium,0.0,Diesel_Manual,Fiat,14.29,1
3986,CAR_003987,Hyundai EON Era Plus,2018,350000.0,15000.0,Petrol,Individual,Manual,First Owner,3987,7,Mid,High,0.0,Petrol_Manual,Hyundai,23.33,0
3987,CAR_003988,Maruti Alto LXi BSIII,2010,190000.0,52000.0,Petrol,Individual,Manual,Second Owner,3988,15,Low,Medium,0.0,Petrol_Manual,Maruti,3.65,0
3988,CAR_003989,Volkswagen Jetta 1.9 TDI Trendline,2009,940000.0,140000.0,Diesel,Individual,Manual,First Owner,3989,16,High,High,0.0,Diesel_Manual,Volkswagen,6.71,1
3989,CAR_003990,Maruti Swift VXI,2013,475000.0,60000.0,Petrol,Individual,Manual,Second Owner,3990,12,Mid,Medium,0.0,Petrol_Manual,Maruti,7.92,0
3990,CAR_003991,Maruti Alto 800 LXI,2018,300000.0,17000.0,Petrol,Individual,Manual,First Owner,3991,7,Low,Low,0.0,Petrol_Manual,Maruti,17.65,0
3991,CAR_003992,Chevrolet Beat LS,2016,250000.0,70000.0,Petrol,Individual,Manual,Third Owner,3992,9,Low,Medium,0.0,Petrol_Manual,Chevrolet,3.57,0
3992,CAR_003993,Maruti Zen LX,2001,93000.0,97000.0,Petrol,Individual,Manual,Fourth & Above Owner,3993,24,Low,High,0.0,Petrol_Manual,Maruti,0.96,0
3993,CAR_003994,Ford Figo Diesel ZXI,2012,290000.0,125000.0,Diesel,Individual,Manual,Second Owner,3994,13,Low,High,0.0,Diesel_Manual,Ford,2.32,1
3994,CAR_003995,Tata Indica GLS BS IV,2010,75000.0,300000.0,Petrol,Individual,Manual,Third Owner,3995,15,Low,Very High,0.0,Petrol_Manual,Tata,0.25,0
3995,CAR_003996,Mahindra XUV500 W8 2WD,2015,4461000.0,120000.0,Diesel,Individual,Manual,Second Owner,3996,10,Low,High,0.0,Diesel_Manual,Mahindra,8.33,1
3996,CAR_003997,Hyundai i20 Asta 1.4 CRDi,2015,550000.0,87000.0,Diesel,Individual,Manual,First Owner,3997,10,Mid,High,0.0,Diesel_Manual,Hyundai,6.32,1
3997,CAR_003998,Fiat Punto 1.3 Emotion,2010,285000.0,75000.0,Diesel,Individual,Manual,Third Owner,3998,15,Low,High,0.0,Diesel_Manual,Fiat,3.8,1
3998,CAR_003999,Tata Indigo Grand Petrol,2014,240000.0,60000.0,Petrol,Individual,Manual,Second Owner,3999,11,Low,Medium,0.0,Petrol_Manual,Tata,4.0,0
3999,CAR_004000,Hyundai Creta 1.6 VTVT S,2015,850000.0,25000.0,Petrol,Individual,Manual,First Owner,4000,10,High,High,0.0,Petrol_Manual,Hyundai,34.0,0
4000,CAR_004001,Hyundai i20 Active 1.2 SX,2016,650000.0,26000.0,Petrol,Individual,Manual,First Owner,4001,9,High,Low,0.0,Petrol_Manual,Hyundai,25.0,0
4001,CAR_004002,Maruti Celerio Green VXI,2017,365000.0,78000.0,CNG,Individual,Manual,First Owner,4002,8,Low,High,0.0,CNG_Manual,Maruti,4.68,0
4002,CAR_004003,Hyundai i20 Active 1.2 SX,2016,4461000.0,26000.0,Petrol,Individual,Manual,First Owner,4003,9,High,Low,0.0,Petrol_Manual,Hyundai,25.0,0
4003,CAR_004004,Chevrolet Sail 1.2 Base,2015,4461000.0,35000.0,Petrol,Individual,Manual,First Owner,4004,10,Low,Medium,0.0,Petrol_Manual,Chevrolet,7.43,0
4004,CAR_004005,Tata Indigo Grand Petrol,2014,250000.0,100000.0,Petrol,Individual,Manual,First Owner,4005,11,Low,High,0.0,Petrol_Manual,Tata,2.5,0
4005,CAR_004006,Tata New Safari 4X2,2007,4461000.0,60000.0,Petrol,Individual,Manual,Second Owner,4006,18,Mid,High,0.0,Petrol_Manual,Tata,6.88,0
4006,CAR_004007,Maruti Wagon R LXI,2004,70000.0,90000.0,Petrol,Individual,Manual,Fourth & Above Owner,4007,21,Low,High,0.0,Petrol_Manual,Maruti,0.78,0
4007,CAR_004008,Chevrolet Sail Hatchback LT ABS,2013,225000.0,80000.0,Diesel,Individual,Manual,First Owner,4008,12,Low,High,0.0,Diesel_Manual,Chevrolet,2.81,1
4008,CAR_004009,Honda City 1.3 EXI,2002,145000.0,100000.0,Petrol,Individual,Manual,First Owner,4009,23,Low,High,0.0,Petrol_Manual,Honda,1.45,0
4009,CAR_004010,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2018,930000.0,20000.0,Diesel,Individual,Manual,First Owner,4010,7,High,Low,0.0,Diesel_Manual,Ford,46.5,1
4010,CAR_004011,Maruti Swift VXI,2019,4461000.0,51000.0,Petrol,Individual,Manual,First Owner,4011,6,Mid,High,0.0,Petrol_Manual,Maruti,10.78,0
4011,CAR_004012,Maruti Alto LX BSIII,2008,85000.0,120000.0,Petrol,Individual,Manual,Second Owner,4012,17,Low,High,0.0,Petrol_Manual,Maruti,0.71,0
4012,CAR_004013,Hyundai EON Era Plus Sports Edition,2014,280000.0,55000.0,Petrol,Individual,Manual,First Owner,4013,11,Low,Medium,0.0,Petrol_Manual,Hyundai,5.09,0
4013,CAR_004014,Hyundai EON Magna Plus,2018,4461000.0,6000.0,Petrol,Individual,Manual,First Owner,4014,7,Mid,Low,0.0,Petrol_Manual,Hyundai,60.0,0
4014,CAR_004015,Maruti Alto LXi,2011,150000.0,40000.0,Petrol,Individual,Manual,Second Owner,4015,14,Low,Medium,0.0,Petrol_Manual,Maruti,3.75,0
4015,CAR_004016,Skoda Rapid 1.6 MPI AT Ambition BSIV,2017,900000.0,15000.0,Petrol,Individual,Automatic,First Owner,4016,8,High,Low,0.0,Petrol_Automatic,Skoda,60.0,0
4016,CAR_004017,Ambassador CLASSIC 1500 DSL AC,2005,120000.0,50000.0,Diesel,Individual,Manual,Second Owner,4017,20,Low,Medium,0.0,Diesel_Manual,Ambassador,2.4,1
4017,CAR_004018,Fiat Linea T Jet Emotion,2010,4461000.0,90000.0,Petrol,Individual,Manual,Second Owner,4018,15,Low,High,0.0,Petrol_Manual,Fiat,2.39,0
4018,CAR_004019,Renault Scala Diesel RxL,2013,550000.0,38500.0,Diesel,Individual,Manual,First Owner,4019,12,Mid,Medium,0.0,Diesel_Manual,Renault,14.29,1
4019,CAR_004020,Tata Nexon 1.2 Revotron XZ Plus Dual Tone,2018,775000.0,35000.0,Petrol,Individual,Manual,First Owner,4020,7,High,Medium,0.0,Petrol_Manual,Tata,22.14,0
4020,CAR_004021,Maruti Gypsy King HT BSIV,2001,409999.0,60000.0,Petrol,Individual,Manual,Third Owner,4021,24,Low,Medium,0.0,Petrol_Manual,Maruti,8.31,0
4021,CAR_004022,Maruti 800 AC Uniq,2005,65000.0,70000.0,Petrol,Individual,Manual,Second Owner,4022,20,Low,Medium,0.0,Petrol_Manual,Maruti,0.93,0
4022,CAR_004023,Maruti Wagon R LXI Minor,2008,200000.0,90000.0,Petrol,Individual,Manual,Second Owner,4023,17,Low,High,0.0,Petrol_Manual,Maruti,2.22,0
4023,CAR_004024,Maruti Wagon R VXI BS IV,2010,240000.0,50000.0,Petrol,Individual,Manual,Second Owner,4024,15,Low,Medium,0.0,Petrol_Manual,Maruti,4.8,0
4024,CAR_004025,Toyota Innova 2.5 VX 8 STR,2012,800000.0,108731.0,Diesel,Individual,Manual,First Owner,4025,13,High,High,0.0,Diesel_Manual,Toyota,7.36,1
4025,CAR_004026,Toyota Innova 2.5 EV Diesel MS 7 Str BSIII,2010,400000.0,120000.0,Diesel,Individual,Manual,First Owner,4026,15,Mid,High,0.0,Diesel_Manual,Toyota,3.33,1
4026,CAR_004027,Maruti Wagon R VXI BSIII,2012,229999.0,40000.0,Petrol,Individual,Manual,Second Owner,4027,13,Low,Medium,0.0,Petrol_Manual,Maruti,5.75,0
4027,CAR_004028,Ford Ecosport 1.5 DV5 MT Titanium,2014,580000.0,120000.0,Diesel,Individual,Manual,First Owner,4028,11,Mid,High,0.0,Diesel_Manual,Ford,4.83,1
4028,CAR_004029,Toyota Corolla H2,2005,175000.0,100000.0,Diesel,Individual,Manual,Fourth & Above Owner,4029,20,Low,High,0.0,Petrol_Manual,Toyota,1.75,0
4029,CAR_004030,Maruti Ritz VDi,2012,250000.0,110000.0,CNG,Individual,Manual,Fourth & Above Owner,4030,13,Low,High,0.0,Diesel_Manual,Maruti,2.27,1
4030,CAR_004031,Hyundai EON Magna Plus,2014,4461000.0,80000.0,Petrol,Individual,Manual,First Owner,4031,11,Low,High,0.0,Petrol_Manual,Hyundai,4.0,0
4031,CAR_004032,Nissan Micra Diesel XV,2013,300000.0,90000.0,Diesel,Individual,Manual,First Owner,4032,12,Low,High,0.0,Diesel_Manual,Nissan,3.33,1
4032,CAR_004033,Mahindra TUV 300 mHAWK100 T8,2017,800000.0,60000.0,Diesel,Individual,Manual,First Owner,4033,8,High,Medium,0.0,Diesel_Manual,Mahindra,13.33,1
4033,CAR_004034,Hyundai Verna SX AT Diesel,2009,4461000.0,75000.0,Diesel,Dealer,Automatic,First Owner,4034,16,Low,High,0.0,Diesel_Automatic,Hyundai,3.27,1
4034,CAR_004035,Hyundai EON Magna Plus,2018,300000.0,20000.0,Petrol,Individual,Manual,First Owner,4035,7,Low,Low,0.0,Petrol_Manual,Hyundai,15.0,0
4035,CAR_004036,Hyundai Xcent 1.2 CRDi E,2017,350000.0,60000.0,Diesel,Individual,Manual,First Owner,4036,8,Mid,High,0.0,Diesel_Manual,Hyundai,4.38,1
4036,CAR_004037,Maruti Zen LX - BS III,2006,70000.0,105700.0,LPG,Individual,Manual,First Owner,4037,19,Low,High,0.0,Petrol_Manual,Maruti,0.66,0
4037,CAR_004038,Maruti Celerio VXI Optional,2017,430999.0,14000.0,Petrol,Dealer,Manual,First Owner,4038,8,Mid,Low,0.0,Petrol_Manual,Maruti,30.79,0
4038,CAR_004039,Chevrolet Spark 1.0,2010,80000.0,70000.0,Petrol,Individual,Manual,First Owner,4039,15,Low,High,0.0,Petrol_Manual,Chevrolet,1.14,0
4039,CAR_004040,Maruti Alto 800 LXI Optional,2019,335000.0,28000.0,Petrol,Dealer,Manual,First Owner,4040,6,Low,High,0.0,Petrol_Manual,Maruti,11.96,0
4040,CAR_004041,Maruti Eeco 7 Seater Standard BSIV,2017,350000.0,51000.0,Petrol,Dealer,Manual,First Owner,4041,8,Mid,Medium,0.0,Petrol_Manual,Maruti,6.86,0
4041,CAR_004042,Maruti Alto 800 LXI Opt BSIV,2018,315000.0,56000.0,Petrol,Dealer,Manual,First Owner,4042,7,Mid,Medium,0.0,Petrol_Manual,Maruti,5.62,0
4042,CAR_004043,Volkswagen Vento 1.5 TDI Highline Plus AT BSIV,2019,1100000.0,35000.0,Diesel,Individual,Automatic,First Owner,4043,6,Premium,Medium,0.0,Diesel_Automatic,Volkswagen,31.43,1
4043,CAR_004044,Maruti Esteem Lxi - BSIII,2005,80000.0,24000.0,Petrol,Individual,Manual,First Owner,4044,20,Low,Low,0.0,Petrol_Manual,Maruti,3.33,0
4044,CAR_004045,Maruti Alto K10 VXI Optional,2014,285000.0,64000.0,Petrol,Dealer,Manual,Second Owner,4045,11,Low,Medium,0.0,Petrol_Manual,Maruti,4.45,0
4045,CAR_004046,Tata Indigo TDI,2009,100000.0,120000.0,Diesel,Individual,Manual,Third Owner,4046,16,Low,High,0.0,Diesel_Manual,Tata,0.83,1
4046,CAR_004047,Maruti Alto LXi,2011,225000.0,37091.0,Petrol,Individual,Manual,First Owner,4047,14,Low,Medium,0.0,Petrol_Manual,Maruti,6.07,0
4047,CAR_004048,Volvo XC 90 D5 Inscription BSIV,2017,4500000.0,80000.0,Diesel,Individual,Automatic,First Owner,4048,8,Premium,High,0.0,Diesel_Automatic,Volvo,56.25,1
4048,CAR_004049,Maruti Alto K10 VXI,2018,320000.0,38900.0,Electric,Individual,Manual,Third Owner,4049,7,Mid,Medium,0.0,Petrol_Manual,Maruti,8.23,0
4049,CAR_004050,Hyundai Creta 1.6 CRDi SX Plus,2016,1249000.0,60000.0,Diesel,Individual,Manual,Third Owner,4050,9,Low,Medium,0.0,Diesel_Manual,Hyundai,20.82,1
4050,CAR_004051,Maruti Wagon R LXI Minor,2008,120000.0,110000.0,Petrol,Individual,Manual,Third Owner,4051,17,Low,High,0.0,Petrol_Manual,Maruti,1.09,0
4051,CAR_004052,Volkswagen Vento Diesel Trendline,2011,200000.0,100000.0,Diesel,Individual,Manual,Second Owner,4052,14,Low,High,0.0,Diesel_Manual,Volkswagen,2.0,1
4052,CAR_004053,Hyundai Creta 1.6 CRDi SX,2016,1151000.0,85000.0,Diesel,Individual,Manual,First Owner,4053,9,Premium,High,0.0,Diesel_Manual,Hyundai,13.54,1
4053,CAR_004054,Tata Indica Vista Quadrajet LS,2012,120000.0,140000.0,Diesel,Individual,Manual,First Owner,4054,13,Low,High,0.0,Diesel_Manual,Tata,0.86,1
4054,CAR_004055,Renault KWID RXT,2017,220000.0,20000.0,Petrol,Individual,Manual,First Owner,4055,8,Low,Low,0.0,Petrol_Manual,Renault,11.0,0
4055,CAR_004056,Tata Nexon 1.2 Revotron XZ Plus Dual Tone,2017,750000.0,15000.0,Petrol,Individual,Manual,First Owner,4056,8,High,Low,0.0,Petrol_Manual,Tata,50.0,0
4056,CAR_004057,Mahindra TUV 300 T10 Dual Tone,2018,784000.0,60000.0,Diesel,Individual,Manual,First Owner,4057,7,High,Medium,0.0,Diesel_Manual,Mahindra,19.6,1
4057,CAR_004058,Honda Jazz 1.2 S i VTEC,2017,500000.0,50000.0,Petrol,Individual,Manual,First Owner,4058,8,Mid,Medium,0.0,Petrol_Manual,Honda,10.0,0
4058,CAR_004059,Hyundai EON Era Plus,2018,350000.0,60000.0,Petrol,Individual,Manual,First Owner,4059,7,Mid,Low,0.0,Petrol_Manual,Hyundai,17.5,0
4059,CAR_004060,Mahindra XUV500 W7 AT BSIV,2018,1400000.0,50000.0,Diesel,Individual,Automatic,First Owner,4060,7,Premium,Medium,0.0,Diesel_Automatic,Mahindra,28.0,1
4060,CAR_004061,Mahindra Thar CRDe AC,2014,600000.0,100000.0,Diesel,Individual,Manual,Second Owner,4061,11,Mid,High,0.0,Diesel_Manual,Mahindra,6.0,1
4061,CAR_004062,Hyundai Grand i10 1.2 Kappa Magna BSIV,2019,525000.0,9400.0,Petrol,Individual,Manual,First Owner,4062,6,Mid,Low,0.0,Petrol_Manual,Hyundai,55.85,0
4062,CAR_004063,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2017,930000.0,12000.0,Diesel,Individual,Manual,First Owner,4063,8,High,Low,0.0,Diesel_Manual,Ford,77.5,1
4063,CAR_004064,Maruti Eeco 5 STR With AC Plus HTR CNG,2017,440000.0,14100.0,CNG,Dealer,Manual,First Owner,4064,8,Mid,Low,0.0,CNG_Manual,Maruti,31.21,0
4064,CAR_004065,Maruti Celerio ZXI,2016,434999.0,21000.0,Petrol,Dealer,Manual,First Owner,4065,9,Mid,High,0.0,Petrol_Manual,Maruti,20.71,0
4065,CAR_004066,Hyundai i10 Magna,2011,250000.0,28000.0,Petrol,Dealer,Manual,First Owner,4066,14,Low,Low,0.0,Petrol_Manual,Hyundai,8.93,0
4066,CAR_004067,Maruti Ertiga SHVS ZDI Plus,2017,940000.0,35000.0,Diesel,Dealer,Manual,First Owner,4067,8,Low,Medium,0.0,Diesel_Manual,Maruti,26.86,1
4067,CAR_004068,Hyundai Verna 1.6 CRDi AT SX,2017,1050000.0,31000.0,Diesel,Dealer,Automatic,First Owner,4068,8,Premium,High,0.0,Diesel_Automatic,Hyundai,33.87,1
4068,CAR_004069,Maruti Ertiga SHVS ZDI,2016,4461000.0,58000.0,Diesel,Dealer,Manual,First Owner,4069,9,High,High,0.0,Diesel_Manual,Maruti,14.22,1
4069,CAR_004070,Maruti Wagon R VXI Minor,2012,229999.0,25000.0,Petrol,Individual,Manual,First Owner,4070,13,Low,Low,0.0,Petrol_Manual,Maruti,9.2,0
4070,CAR_004071,Hyundai i20 1.2 Asta,2011,375000.0,72000.0,Petrol,Dealer,Manual,First Owner,4071,14,Mid,High,0.0,Petrol_Manual,Hyundai,5.21,0
4071,CAR_004072,Maruti Ertiga ZDI,2013,640000.0,89000.0,Diesel,Dealer,Manual,First Owner,4072,12,High,High,0.0,Diesel_Manual,Maruti,7.19,1
4072,CAR_004073,Toyota Innova 2.5 V Diesel 8-seater,2012,4461000.0,160000.0,Diesel,Dealer,Manual,First Owner,4073,13,High,Very High,0.0,Diesel_Manual,Toyota,5.0,1
4073,CAR_004074,Ford Figo Petrol ZXI,2012,130000.0,30000.0,Petrol,Individual,Manual,Third Owner,4074,13,Low,Low,0.0,Petrol_Manual,Ford,4.33,0
4074,CAR_004075,Hyundai i10 Sportz 1.1L,2014,300000.0,37555.0,Petrol,Dealer,Manual,First Owner,4075,11,Low,Medium,0.0,Petrol_Manual,Hyundai,7.99,0
4075,CAR_004076,Maruti Wagon R VXI BS IV,2018,440000.0,14000.0,Petrol,Dealer,Manual,First Owner,4076,7,Mid,Low,0.0,Petrol_Manual,Maruti,31.43,0
4076,CAR_004077,Maruti Ertiga VXI CNG,2014,595000.0,56600.0,CNG,Dealer,Manual,First Owner,4077,11,Mid,Medium,0.0,CNG_Manual,Maruti,10.51,0
4077,CAR_004078,Mahindra Scorpio EX,2013,450000.0,110000.0,Diesel,Individual,Manual,First Owner,4078,12,Mid,High,0.0,Diesel_Manual,Mahindra,4.09,1
4078,CAR_004079,Maruti Vitara Brezza VDi Option,2017,690000.0,50000.0,Diesel,Individual,Manual,First Owner,4079,8,High,Medium,0.0,Diesel_Manual,Maruti,13.8,1
4079,CAR_004080,Maruti Alto 800 VXI,2019,4461000.0,5000.0,Petrol,Individual,Manual,First Owner,4080,6,Low,Low,0.0,Petrol_Manual,Maruti,60.0,0
4080,CAR_004081,Chevrolet Sail 1.2 Base,2014,170000.0,100000.0,Petrol,Individual,Manual,Second Owner,4081,11,Low,High,0.0,Petrol_Manual,Chevrolet,1.7,0
4081,CAR_004082,Toyota Etios Liva GD SP,2012,280000.0,120000.0,Diesel,Individual,Manual,First Owner,4082,13,Low,High,0.0,Diesel_Manual,Toyota,2.33,1
4082,CAR_004083,Maruti Swift VXI BSIII,2009,220000.0,67580.0,Petrol,Individual,Manual,Second Owner,4083,16,Low,High,0.0,Petrol_Manual,Maruti,3.26,0
4083,CAR_004084,Hyundai Santro Xing GLS,2009,140000.0,100000.0,Petrol,Individual,Manual,Third Owner,4084,16,Low,High,0.0,Petrol_Manual,Hyundai,1.4,0
4084,CAR_004085,Maruti Esteem Lxi,2004,45000.0,60000.0,Petrol,Individual,Manual,First Owner,4085,21,Low,Medium,0.0,Petrol_Manual,Maruti,0.75,0
4085,CAR_004086,Ford Figo Diesel Titanium,2014,200000.0,125000.0,Diesel,Individual,Manual,Second Owner,4086,11,Low,High,0.0,Diesel_Manual,Ford,1.6,1
4086,CAR_004087,Mahindra Scorpio 2.6 Turbo 7 Str,2008,325000.0,120000.0,Diesel,Individual,Manual,Second Owner,4087,17,Mid,High,0.0,Diesel_Manual,Mahindra,2.71,1
4087,CAR_004088,Hyundai Verna CRDi SX,2010,300000.0,60000.0,Diesel,Individual,Manual,Second Owner,4088,15,Low,High,0.0,Diesel_Manual,Hyundai,2.31,1
4088,CAR_004089,Maruti 800 AC,2009,120000.0,250000.0,Petrol,Individual,Manual,Second Owner,4089,16,Low,Very High,0.0,Petrol_Manual,Maruti,0.48,0
4089,CAR_004090,Hyundai Creta 1.6 CRDi SX,2016,535000.0,52600.0,Diesel,Individual,Manual,First Owner,4090,9,Mid,Medium,0.0,Diesel_Manual,Hyundai,10.17,1
4090,CAR_004091,Honda City i DTEC S,2014,520000.0,82000.0,Diesel,Individual,Manual,First Owner,4091,11,Low,High,0.0,Diesel_Manual,Honda,6.34,1
4091,CAR_004092,Hyundai Grand i10 1.2 Kappa Magna AT,2017,550000.0,19890.0,Petrol,Dealer,Automatic,First Owner,4092,8,Low,Low,0.0,Petrol_Automatic,Hyundai,27.65,0
4092,CAR_004093,Mahindra Verito Vibe 1.5 dCi D6,2015,280000.0,40000.0,Diesel,Individual,Manual,First Owner,4093,10,Low,Medium,0.0,Diesel_Manual,Mahindra,7.0,1
4093,CAR_004094,Nissan Sunny XL D,2013,225000.0,78000.0,Diesel,Dealer,Manual,First Owner,4094,12,Low,High,0.0,Diesel_Manual,Nissan,2.88,1
4094,CAR_004095,Ford EcoSport 1.5 Petrol Titanium Plus AT BSIV,2015,750000.0,48238.0,Petrol,Trustmark Dealer,Automatic,First Owner,4095,10,High,Medium,0.0,Petrol_Automatic,Ford,15.55,0
4095,CAR_004096,Hyundai i20 1.2 Magna Executive,2016,4461000.0,38365.0,Petrol,Dealer,Manual,First Owner,4096,9,Mid,Medium,0.0,Petrol_Manual,Hyundai,13.95,0
4096,CAR_004097,Mahindra XUV500 W10 1.99 mHawk,2016,1250000.0,23670.0,Diesel,Trustmark Dealer,Manual,First Owner,4097,9,Premium,Low,0.0,Diesel_Manual,Mahindra,52.81,1
4097,CAR_004098,Maruti Baleno Zeta 1.2,2017,650000.0,49834.0,Diesel,Trustmark Dealer,Manual,First Owner,4098,8,High,Medium,0.0,Petrol_Manual,Maruti,13.04,0
4098,CAR_004099,Honda Brio S MT,2013,295000.0,57353.0,Petrol,Trustmark Dealer,Manual,First Owner,4099,12,Low,High,0.0,Petrol_Manual,Honda,5.14,0
4099,CAR_004100,Maruti Swift DDiS VDI,2015,480000.0,68308.0,Diesel,Trustmark Dealer,Manual,First Owner,4100,10,Mid,Medium,0.0,Diesel_Manual,Maruti,7.03,1
4100,CAR_004101,Honda City i DTEC SV,2015,575000.0,63240.0,Diesel,Trustmark Dealer,Manual,First Owner,4101,10,Mid,High,0.0,Diesel_Manual,Honda,9.09,1
4101,CAR_004102,Mahindra Scorpio 1.99 S4,2016,509999.0,60000.0,CNG,Individual,Manual,First Owner,4102,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.5,1
4102,CAR_004103,Maruti Alto LXi,2008,69000.0,100000.0,Petrol,Individual,Manual,First Owner,4103,17,Low,High,0.0,Petrol_Manual,Maruti,0.69,0
4103,CAR_004104,Tata Manza Aura Safire BS IV,2013,200000.0,60000.0,Petrol,Individual,Manual,First Owner,4104,12,Low,Medium,0.0,Petrol_Manual,Tata,3.33,0
4104,CAR_004105,Toyota Innova 2.5 GX 8 STR BSIV,2012,449000.0,90000.0,Diesel,Individual,Manual,First Owner,4105,13,Mid,High,0.0,Diesel_Manual,Toyota,4.99,1
4105,CAR_004106,Tata Harrier XE,2020,426000.0,1000.0,Diesel,Individual,Manual,First Owner,4106,5,Mid,Low,0.0,Diesel_Manual,Tata,426.0,1
4106,CAR_004107,Ford Ecosport 1.5 Petrol Ambiente,2017,650000.0,60000.0,Petrol,Individual,Manual,First Owner,4107,8,High,Low,0.0,Petrol_Manual,Ford,108.33,0
4107,CAR_004108,Tata Tiago 1.05 Revotorq XM,2017,260000.0,50000.0,Diesel,Individual,Manual,First Owner,4108,8,Low,Medium,0.0,Diesel_Manual,Tata,5.2,1
4108,CAR_004109,Mahindra Bolero Power Plus Plus AC BSIV PS,2015,295000.0,90000.0,Diesel,Individual,Manual,Third Owner,4109,10,Low,High,0.0,Diesel_Manual,Mahindra,3.28,1
4109,CAR_004110,Maruti Eeco 5 Seater AC BSIV,2017,260000.0,30000.0,Petrol,Dealer,Manual,First Owner,4110,8,Low,Low,0.0,Petrol_Manual,Maruti,8.67,0
4110,CAR_004111,Tata Indica Vista Quadrajet LS,2012,120000.0,140000.0,Diesel,Individual,Manual,First Owner,4111,13,Low,High,0.0,Diesel_Manual,Tata,0.86,1
4111,CAR_004112,Hyundai Verna SX,2007,155000.0,65000.0,Petrol,Dealer,Manual,First Owner,4112,18,Low,Medium,0.0,Petrol_Manual,Hyundai,2.38,0
4112,CAR_004113,Honda Jazz 1.5 VX i DTEC,2015,450000.0,60000.0,Diesel,Individual,Manual,First Owner,4113,10,Mid,Medium,0.0,Diesel_Manual,Honda,11.84,1
4113,CAR_004114,Tata New Safari DICOR 2.2 GX 4x2 BS IV,2012,320000.0,80000.0,Diesel,Individual,Manual,First Owner,4114,13,Mid,High,0.0,Diesel_Manual,Tata,4.0,1
4114,CAR_004115,Hyundai Verna i (Petrol),2007,123000.0,50000.0,Petrol,Individual,Manual,Second Owner,4115,18,Low,Medium,0.0,Petrol_Manual,Hyundai,2.46,0
4115,CAR_004116,Hyundai Santro Xing XO,2006,4461000.0,110000.0,LPG,Individual,Manual,First Owner,4116,19,Low,High,0.0,Petrol_Manual,Hyundai,0.73,0
4116,CAR_004117,Maruti SX4 Celebration Petrol,2012,220000.0,90000.0,Petrol,Individual,Manual,Second Owner,4117,13,Low,High,0.0,Petrol_Manual,Maruti,2.44,0
4117,CAR_004118,Hyundai i20 Active S Diesel,2018,650000.0,37000.0,Diesel,Dealer,Manual,First Owner,4118,7,High,Medium,0.0,Diesel_Manual,Hyundai,17.57,1
4118,CAR_004119,Maruti Alto 800 LXI,2018,285000.0,30000.0,Petrol,Dealer,Manual,First Owner,4119,7,Low,High,0.0,Petrol_Manual,Maruti,9.5,0
4119,CAR_004120,Honda Amaze V CVT Petrol BSIV,2018,725000.0,26000.0,Petrol,Dealer,Automatic,First Owner,4120,7,High,Low,0.0,Petrol_Automatic,Honda,27.88,0
4120,CAR_004121,Tata Indica Vista Aqua 1.2 Safire BSIV,2010,97000.0,128000.0,Electric,Individual,Manual,First Owner,4121,15,Low,High,0.0,Petrol_Manual,Tata,0.76,0
4121,CAR_004122,Maruti Ciaz ZXi,2016,625000.0,30000.0,Petrol,Dealer,Manual,First Owner,4122,9,High,Low,0.0,Petrol_Manual,Maruti,20.83,0
4122,CAR_004123,Hyundai i20 Active SX Petrol,2016,550000.0,23000.0,Petrol,Dealer,Manual,First Owner,4123,9,Mid,Low,0.0,Petrol_Manual,Hyundai,23.91,0
4123,CAR_004124,Tata Indigo CR4,2012,130000.0,120000.0,Diesel,Individual,Manual,First Owner,4124,13,Low,High,0.0,Diesel_Manual,Tata,1.08,1
4124,CAR_004125,Hyundai i10 Magna,2012,240000.0,40000.0,Petrol,Dealer,Manual,First Owner,4125,13,Low,Medium,0.0,Petrol_Manual,Hyundai,6.0,0
4125,CAR_004126,Nissan Terrano XL 85 PS,2014,600000.0,60000.0,Diesel,Dealer,Manual,First Owner,4126,11,Mid,Low,0.0,Diesel_Manual,Nissan,22.22,1
4126,CAR_004127,Tata New Safari DICOR 2.2 EX 4x2,2012,350000.0,65000.0,Diesel,Dealer,Manual,First Owner,4127,13,Mid,Medium,0.0,Diesel_Manual,Tata,5.38,1
4127,CAR_004128,Hyundai EON Magna Plus,2015,245000.0,32000.0,Petrol,Dealer,Manual,First Owner,4128,10,Low,Medium,0.0,Petrol_Manual,Hyundai,7.66,0
4128,CAR_004129,Maruti Swift 1.3 VXi,2009,199000.0,52536.0,Petrol,Individual,Manual,First Owner,4129,16,Low,Medium,0.0,Petrol_Manual,Maruti,3.79,0
4129,CAR_004130,Mahindra Xylo E4,2011,330000.0,100000.0,Diesel,Individual,Manual,Second Owner,4130,14,Mid,High,0.0,Diesel_Manual,Mahindra,3.3,1
4130,CAR_004131,Maruti Ritz VXI,2009,150000.0,80000.0,Petrol,Individual,Manual,First Owner,4131,16,Low,High,0.0,Petrol_Manual,Maruti,1.88,0
4131,CAR_004132,Hyundai Creta 1.6 CRDi SX,2016,927999.0,90000.0,Diesel,Individual,Manual,First Owner,4132,9,High,High,0.0,Diesel_Manual,Hyundai,10.31,1
4132,CAR_004133,Maruti Zen LXI,2003,70000.0,60000.0,Petrol,Individual,Manual,Fourth & Above Owner,4133,22,Low,Medium,0.0,Petrol_Manual,Maruti,1.17,0
4133,CAR_004134,Hyundai Xcent 1.2 CRDi SX,2015,399000.0,55000.0,Diesel,Dealer,Manual,First Owner,4134,10,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.25,1
4134,CAR_004135,Maruti Swift VDI Optional,2015,4461000.0,90000.0,Diesel,Individual,Manual,First Owner,4135,10,Mid,High,0.0,Diesel_Manual,Maruti,3.89,1
4135,CAR_004136,Mahindra Scorpio 2.6 CRDe,2006,125000.0,50000.0,Diesel,Individual,Manual,Second Owner,4136,19,Low,Medium,0.0,Diesel_Manual,Mahindra,2.5,1
4136,CAR_004137,Hyundai Santro Xing XS,2008,4461000.0,90000.0,Diesel,Individual,Manual,Second Owner,4137,17,Low,High,0.0,Petrol_Manual,Hyundai,1.67,0
4137,CAR_004138,Hyundai EON Era Plus,2015,250000.0,25000.0,Petrol,Individual,Manual,First Owner,4138,10,Low,Low,0.0,Petrol_Manual,Hyundai,10.0,0
4138,CAR_004139,Hyundai Verna 1.6 SX CRDi (O),2014,675000.0,61000.0,Diesel,Dealer,Manual,First Owner,4139,11,High,Medium,0.0,Diesel_Manual,Hyundai,11.07,1
4139,CAR_004140,Maruti S-Cross Alpha DDiS 200 SH,2018,950000.0,60000.0,CNG,Individual,Manual,First Owner,4140,7,High,Low,0.0,Diesel_Manual,Maruti,31.67,1
4140,CAR_004141,Maruti Eeco 5 STR With AC Plus HTR CNG,2018,409999.0,45000.0,CNG,Individual,Manual,First Owner,4141,7,Mid,Medium,0.0,CNG_Manual,Maruti,9.11,0
4141,CAR_004142,Fiat Linea Emotion (Diesel),2011,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,4142,14,Low,Medium,0.0,Diesel_Manual,Fiat,3.75,1
4142,CAR_004143,Honda Civic 1.8 V MT,2008,350000.0,75000.0,Petrol,Individual,Manual,Second Owner,4143,17,Mid,High,0.0,Petrol_Manual,Honda,4.67,0
4143,CAR_004144,Tata Indica LSI,2005,51111.0,140000.0,Petrol,Individual,Manual,Third Owner,4144,20,Low,High,0.0,Petrol_Manual,Tata,0.37,0
4144,CAR_004145,Chevrolet Sail 1.3 LS,2013,290000.0,56000.0,Diesel,Dealer,Manual,First Owner,4145,12,Low,Medium,0.0,Diesel_Manual,Chevrolet,5.18,1
4145,CAR_004146,Toyota Camry Hybrid,2006,310000.0,62000.0,Electric,Dealer,Automatic,Second Owner,4146,19,Mid,Medium,0.0,Electric_Automatic,Toyota,5.0,0
4146,CAR_004147,Hyundai Verna CRDi SX,2007,212000.0,58000.0,Diesel,Dealer,Manual,Second Owner,4147,18,Low,Medium,0.0,Diesel_Manual,Hyundai,3.66,1
4147,CAR_004148,Skoda Octavia Elegance 2.0 TDI AT,2013,900000.0,49000.0,Diesel,Dealer,Manual,Second Owner,4148,12,High,Medium,0.0,Diesel_Automatic,Skoda,18.37,1
4148,CAR_004149,Chevrolet Beat Diesel LT,2011,221000.0,56000.0,Diesel,Dealer,Manual,Second Owner,4149,14,Low,High,0.0,Diesel_Manual,Chevrolet,3.95,1
4149,CAR_004150,Toyota Innova 2.5 G (Diesel) 7 Seater,2013,675000.0,60000.0,Diesel,Dealer,Manual,First Owner,4150,12,High,High,0.0,Diesel_Manual,Toyota,9.25,1
4150,CAR_004151,Honda Amaze S i-Dtech,2014,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,4151,11,Low,Medium,0.0,Diesel_Manual,Honda,6.1,1
4151,CAR_004152,Maruti SX4 Zxi BSIII,2008,350000.0,10000.0,Petrol,Individual,Manual,First Owner,4152,17,Mid,Low,0.0,Petrol_Manual,Maruti,35.0,0
4152,CAR_004153,Hyundai Accent Executive CNG,2010,140000.0,110000.0,CNG,Individual,Manual,First Owner,4153,15,Low,High,0.0,CNG_Manual,Hyundai,1.27,0
4153,CAR_004154,Skoda Octavia Ambiente 1.9 TDI MT,2002,4461000.0,116000.0,Diesel,Individual,Manual,Second Owner,4154,23,Low,High,0.0,Diesel_Manual,Skoda,0.78,1
4154,CAR_004155,Honda City Edge Edition Diesel SV,2018,830000.0,40000.0,Diesel,Individual,Manual,First Owner,4155,7,High,Medium,0.0,Diesel_Manual,Honda,20.75,1
4155,CAR_004156,Maruti Wagon R LXI Minor,2009,180000.0,30000.0,Petrol,Individual,Manual,First Owner,4156,16,Low,Low,0.0,Petrol_Manual,Maruti,6.0,0
4156,CAR_004157,Maruti Swift Vdi BSIII,2009,210000.0,109000.0,Diesel,Individual,Manual,Second Owner,4157,16,Low,High,0.0,Diesel_Manual,Maruti,1.93,1
4157,CAR_004158,Maruti SX4 S Cross DDiS 320 Zeta,2017,700000.0,90000.0,Diesel,Individual,Manual,First Owner,4158,8,High,High,0.0,Diesel_Manual,Maruti,7.78,1
4158,CAR_004159,Tata Nano Cx BSIII,2014,45000.0,7000.0,Petrol,Individual,Manual,Second Owner,4159,11,Low,High,0.0,Petrol_Manual,Tata,6.43,0
4159,CAR_004160,Tata Indigo LX,2012,125000.0,60000.0,LPG,Individual,Manual,First Owner,4160,13,Low,High,0.0,Diesel_Manual,Tata,1.04,1
4160,CAR_004161,Hyundai Verna 1.6 VTVT S,2017,530000.0,64916.0,Electric,Individual,Manual,First Owner,4161,8,Mid,Medium,0.0,Petrol_Manual,Hyundai,8.16,0
4161,CAR_004162,Ford Ecosport 1.5 DV5 MT Trend,2018,800000.0,37161.0,Diesel,Dealer,Manual,First Owner,4162,7,High,Medium,0.0,Diesel_Manual,Ford,21.53,1
4162,CAR_004163,Maruti Baleno Zeta 1.2,2016,580000.0,15000.0,Petrol,Individual,Manual,First Owner,4163,9,Mid,Low,0.0,Petrol_Manual,Maruti,38.67,0
4163,CAR_004164,Audi Q3 2.0 TDI Quattro Premium Plus,2012,4461000.0,118000.0,Diesel,Individual,Automatic,Second Owner,4164,13,Premium,High,0.0,Diesel_Automatic,Audi,11.86,1
4164,CAR_004165,Renault KWID 1.0 RXL,2018,300000.0,50852.0,Petrol,Individual,Manual,First Owner,4165,7,Low,Medium,0.0,Petrol_Manual,Renault,5.9,0
4165,CAR_004166,Renault KWID RXT Optional,2016,300000.0,30000.0,Petrol,Individual,Manual,First Owner,4166,9,Low,Low,0.0,Petrol_Manual,Renault,10.0,0
4166,CAR_004167,Maruti Ciaz ZDi SHVS,2015,700000.0,77000.0,Diesel,Individual,Manual,First Owner,4167,10,High,High,0.0,Diesel_Manual,Maruti,9.09,1
4167,CAR_004168,Nissan Sunny XL,2012,4461000.0,60000.0,Petrol,Individual,Manual,Third Owner,4168,13,Low,Medium,0.0,Petrol_Manual,Nissan,3.57,0
4168,CAR_004169,Chevrolet Tavera Neo 3 9 Str BSIII,2016,500000.0,120000.0,Diesel,Individual,Manual,First Owner,4169,9,Mid,High,0.0,Diesel_Manual,Chevrolet,4.17,1
4169,CAR_004170,Honda Accord 2.4 AT,2009,428000.0,60000.0,Petrol,Individual,Automatic,Third Owner,4170,16,Mid,Medium,0.0,Petrol_Automatic,Honda,10.7,0
4170,CAR_004171,Tata Harrier XZ BSIV,2019,1400000.0,33000.0,CNG,Individual,Manual,First Owner,4171,6,Low,Medium,0.0,Diesel_Manual,Tata,42.42,1
4171,CAR_004172,Hyundai EON Era Plus,2013,219000.0,53500.0,Petrol,Individual,Manual,Third Owner,4172,12,Low,Medium,0.0,Petrol_Manual,Hyundai,4.09,0
4172,CAR_004173,Maruti Ertiga SHVS LDI,2017,500000.0,40000.0,Diesel,Individual,Manual,First Owner,4173,8,Mid,Medium,0.0,Diesel_Manual,Maruti,12.5,1
4173,CAR_004174,Honda City i DTEC S,2014,520000.0,60000.0,LPG,Individual,Manual,First Owner,4174,11,Mid,High,0.0,Diesel_Manual,Honda,6.34,1
4174,CAR_004175,Mahindra Scorpio 1.99 S4,2016,509999.0,60000.0,Electric,Individual,Manual,First Owner,4175,9,Mid,Medium,0.0,Diesel_Manual,Mahindra,8.5,1
4175,CAR_004176,Maruti Baleno Alpha 1.2,2019,556000.0,24000.0,Petrol,Individual,Manual,First Owner,4176,6,Mid,Low,0.0,Petrol_Manual,Maruti,23.17,0
4176,CAR_004177,Tata Indigo Grand Dicor,2014,225000.0,120000.0,Diesel,Individual,Manual,First Owner,4177,11,Low,High,0.0,Diesel_Manual,Tata,1.88,1
4177,CAR_004178,Honda Amaze EX i-Dtech,2013,325000.0,65000.0,Diesel,Dealer,Manual,First Owner,4178,12,Mid,Medium,0.0,Diesel_Manual,Honda,5.0,1
4178,CAR_004179,Maruti Swift VDI,2013,365000.0,65000.0,Diesel,Dealer,Manual,First Owner,4179,12,Mid,Medium,0.0,Diesel_Manual,Maruti,5.62,1
4179,CAR_004180,Hyundai Creta 1.6 CRDi SX Plus,2015,850000.0,60000.0,Diesel,Dealer,Manual,First Owner,4180,10,Low,High,0.0,Diesel_Manual,Hyundai,10.12,1
4180,CAR_004181,Toyota Etios GD SP,2013,350000.0,75000.0,Diesel,Dealer,Manual,First Owner,4181,12,Mid,High,0.0,Diesel_Manual,Toyota,4.67,1
4181,CAR_004182,Maruti Swift VDI,2007,4461000.0,60000.0,Diesel,Dealer,Manual,First Owner,4182,18,Low,Medium,0.0,Diesel_Manual,Maruti,4.5,1
4182,CAR_004183,Maruti Alto LX,2011,4461000.0,50000.0,Petrol,Individual,Manual,First Owner,4183,14,Low,Medium,0.0,Petrol_Manual,Maruti,2.7,0
4183,CAR_004184,Maruti Wagon R LXI Minor,2007,140000.0,49000.0,Petrol,Dealer,Manual,First Owner,4184,18,Low,Medium,0.0,Petrol_Manual,Maruti,2.86,0
4184,CAR_004185,Maruti SX4 S Cross DDiS 320 Delta,2016,665000.0,560000.0,Diesel,Dealer,Manual,First Owner,4185,9,High,Very High,0.0,Diesel_Manual,Maruti,1.19,1
4185,CAR_004186,Renault KWID RXT,2015,275000.0,14365.0,Petrol,Dealer,Manual,First Owner,4186,10,Low,High,0.0,Petrol_Manual,Renault,19.14,0
4186,CAR_004187,Toyota Fortuner 2.8 4WD AT BSIV,2017,2750000.0,41000.0,Diesel,Individual,Automatic,First Owner,4187,8,Premium,Medium,0.0,Diesel_Automatic,Toyota,67.07,1
4187,CAR_004188,Hyundai Verna 1.6 SX,2013,484999.0,65000.0,Diesel,Dealer,Manual,First Owner,4188,12,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.46,1
4188,CAR_004189,Maruti Ciaz VDi Option SHVS,2015,565000.0,60000.0,Diesel,Dealer,Manual,First Owner,4189,10,Mid,Medium,0.0,Diesel_Manual,Maruti,8.69,1
4189,CAR_004190,Maruti Swift Dzire VDI,2013,425000.0,60000.0,Diesel,Dealer,Manual,First Owner,4190,12,Mid,Medium,0.0,Diesel_Manual,Maruti,6.96,1
4190,CAR_004191,Toyota Innova 2.5 VX (Diesel) 8 Seater,2013,925000.0,75000.0,Diesel,Individual,Manual,First Owner,4191,12,High,High,0.0,Diesel_Manual,Toyota,12.33,1
4191,CAR_004192,Mahindra Scorpio VLX 2WD AIRBAG SE BSIV,2012,565000.0,72000.0,Diesel,Dealer,Manual,First Owner,4192,13,Mid,High,0.0,Diesel_Manual,Mahindra,7.85,1
4192,CAR_004193,Renault Duster 110PS Diesel RxL,2014,525000.0,65000.0,Diesel,Dealer,Manual,First Owner,4193,11,Mid,Medium,0.0,Diesel_Manual,Renault,8.08,1
4193,CAR_004194,Honda City 1.5 S MT,2010,409999.0,60000.0,Petrol,Individual,Manual,First Owner,4194,15,Mid,Medium,0.0,Petrol_Manual,Honda,6.83,0
4194,CAR_004195,Tata Tigor 1.2 Revotron XZ Option,2018,570000.0,5000.0,Petrol,Individual,Manual,First Owner,4195,7,Mid,Low,0.0,Petrol_Manual,Tata,114.0,0
4195,CAR_004196,Maruti SX4 Zxi BSIII,2007,160000.0,80000.0,CNG,Individual,Manual,Third Owner,4196,18,Low,High,0.0,Petrol_Manual,Maruti,2.0,0
4196,CAR_004197,Maruti Ciaz ZDi Plus SHVS,2017,749000.0,51500.0,Diesel,Individual,Manual,Second Owner,4197,8,Low,Medium,0.0,Diesel_Manual,Maruti,14.54,1
4197,CAR_004198,Maruti Swift Dzire VDi,2010,300000.0,120000.0,Diesel,Individual,Manual,Third Owner,4198,15,Low,High,0.0,Diesel_Manual,Maruti,2.5,1
4198,CAR_004199,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2014,550000.0,79800.0,Diesel,Dealer,Manual,First Owner,4199,11,Mid,High,0.0,Diesel_Manual,Ford,6.89,1
4199,CAR_004200,Ford EcoSport 1.5 Diesel Titanium Plus BSIV,2015,550000.0,97000.0,Diesel,Dealer,Manual,Second Owner,4200,10,Mid,High,0.0,Diesel_Manual,Ford,5.67,1
4200,CAR_004201,Ford Ecosport 1.5 Diesel Titanium Plus,2019,1250000.0,6590.0,Diesel,Dealer,Manual,First Owner,4201,6,Premium,Low,0.0,Diesel_Manual,Ford,189.68,1
4201,CAR_004202,Ford Figo Aspire 1.5 TDCi Titanium,2017,700000.0,49957.0,Diesel,Dealer,Manual,First Owner,4202,8,High,Medium,0.0,Diesel_Manual,Ford,14.01,1
4202,CAR_004203,Ford Figo 1.5 Sports Edition MT,2017,600000.0,43235.0,Diesel,Dealer,Manual,First Owner,4203,8,Mid,Medium,0.0,Diesel_Manual,Ford,13.88,1
4203,CAR_004204,Ford EcoSport 1.5 TDCi Titanium Plus BE BSIV,2018,925000.0,50699.0,Diesel,Dealer,Manual,First Owner,4204,7,High,Medium,0.0,Diesel_Manual,Ford,18.24,1
4204,CAR_004205,Ford Endeavour 3.2 Titanium AT 4X4,2016,1800000.0,126000.0,Diesel,Dealer,Automatic,First Owner,4205,9,Premium,High,0.0,Diesel_Automatic,Ford,14.29,1
4205,CAR_004206,Ford Figo Aspire 1.5 TDCi Titanium Plus,2015,459999.0,140730.0,Diesel,Dealer,Manual,Second Owner,4206,10,Mid,High,0.0,Diesel_Manual,Ford,3.27,1
4206,CAR_004207,Hyundai i20 Active 1.2 SX,2015,565000.0,46000.0,Petrol,Individual,Manual,First Owner,4207,10,Mid,High,0.0,Petrol_Manual,Hyundai,12.28,0
4207,CAR_004208,Hyundai EON Era Plus,2013,170000.0,80000.0,LPG,Individual,Manual,Third Owner,4208,12,Low,High,0.0,Petrol_Manual,Hyundai,2.12,0
4208,CAR_004209,Toyota Qualis FS B3,2001,4461000.0,256000.0,Diesel,Dealer,Manual,First Owner,4209,24,Low,Very High,0.0,Diesel_Manual,Toyota,0.59,1
4209,CAR_004210,Mahindra Scorpio VLX 2WD BSIII,2008,4461000.0,120000.0,Electric,Individual,Manual,First Owner,4210,17,Low,High,0.0,Diesel_Manual,Mahindra,2.46,1
4210,CAR_004211,Hyundai i20 Active 1.4 SX,2018,819999.0,40000.0,Diesel,Individual,Manual,First Owner,4211,7,High,Medium,0.0,Diesel_Manual,Hyundai,20.5,1
4211,CAR_004212,Hyundai Elite i20 Diesel Asta Option,2019,819999.0,40000.0,Diesel,Individual,Manual,First Owner,4212,6,High,Medium,0.0,Diesel_Manual,Hyundai,20.5,1
4212,CAR_004213,Maruti Ritz VXi,2015,320000.0,40000.0,Petrol,Individual,Manual,First Owner,4213,10,Mid,Medium,0.0,Petrol_Manual,Maruti,8.0,0
4213,CAR_004214,Ford Ecosport 1.5 DV5 MT Titanium,2015,550000.0,135000.0,Diesel,Individual,Manual,Second Owner,4214,10,Mid,High,0.0,Diesel_Manual,Ford,4.07,1
4214,CAR_004215,Mahindra Bolero 2011-2019 SLX,2015,550000.0,50000.0,Diesel,Individual,Manual,First Owner,4215,10,Mid,Medium,0.0,Diesel_Manual,Mahindra,11.0,1
4215,CAR_004216,Volkswagen Ameo 1.5 TDI Comfortline,2017,4461000.0,98000.0,Diesel,Individual,Manual,First Owner,4216,8,Low,High,0.0,Diesel_Manual,Volkswagen,6.02,1
4216,CAR_004217,Toyota Innova 2.5 VX (Diesel) 8 Seater,2014,1050000.0,70000.0,Diesel,Individual,Manual,First Owner,4217,11,Premium,Medium,0.0,Diesel_Manual,Toyota,15.0,1
4217,CAR_004218,Honda City i DTec V,2017,950000.0,50000.0,Diesel,Individual,Manual,First Owner,4218,8,High,Medium,0.0,Diesel_Manual,Honda,19.0,1
4218,CAR_004219,Honda City i-DTEC VX,2017,4461000.0,30000.0,Diesel,Individual,Manual,First Owner,4219,8,High,Low,0.0,Diesel_Manual,Honda,33.33,1
4219,CAR_004220,Nissan Terrano XE 85 PS,2013,500000.0,120000.0,Diesel,Individual,Manual,First Owner,4220,12,Mid,High,0.0,Diesel_Manual,Nissan,4.17,1
4220,CAR_004221,Fiat Punto EVO 1.3 Emotion,2017,450000.0,70000.0,Diesel,Individual,Manual,First Owner,4221,8,Mid,Medium,0.0,Diesel_Manual,Fiat,6.43,1
4221,CAR_004222,Maruti Swift Dzire VDI Optional,2017,720000.0,35000.0,Diesel,Individual,Manual,First Owner,4222,8,High,Medium,0.0,Diesel_Manual,Maruti,20.57,1
4222,CAR_004223,Maruti Celerio VDi,2015,390000.0,60000.0,Diesel,Individual,Manual,Second Owner,4223,10,Mid,Medium,0.0,Diesel_Manual,Maruti,5.57,1
4223,CAR_004224,Mahindra Thar 4X2,2014,500000.0,35000.0,Diesel,Individual,Manual,First Owner,4224,11,Mid,Medium,0.0,Diesel_Manual,Mahindra,14.29,1
4224,CAR_004225,Toyota Fortuner 2.7 2WD AT,2016,2500000.0,70000.0,Petrol,Individual,Automatic,Second Owner,4225,9,Premium,Medium,0.0,Petrol_Automatic,Toyota,35.71,0
4225,CAR_004226,Hyundai i20 1.4 Asta Option,2017,780000.0,50000.0,Diesel,Individual,Manual,First Owner,4226,8,High,Medium,0.0,Diesel_Manual,Hyundai,15.6,1
4226,CAR_004227,Maruti Swift Dzire ZDI,2015,484999.0,90000.0,Diesel,Individual,Manual,Second Owner,4227,10,Mid,High,0.0,Diesel_Manual,Maruti,5.39,1
4227,CAR_004228,Volkswagen Vento Diesel Highline,2012,420000.0,90000.0,Diesel,Individual,Manual,Second Owner,4228,13,Mid,High,0.0,Diesel_Manual,Volkswagen,4.67,1
4228,CAR_004229,Mahindra XUV500 W10 AWD,2015,1225000.0,60000.0,Diesel,Individual,Manual,First Owner,4229,10,Premium,Medium,0.0,Diesel_Manual,Mahindra,17.5,1
4229,CAR_004230,Tata Indica Vista TDI LX,2015,350000.0,50000.0,Diesel,Individual,Manual,Second Owner,4230,10,Mid,Medium,0.0,Diesel_Manual,Tata,7.0,1
4230,CAR_004231,Mahindra Verito 1.5 D4 BSIV,2015,350000.0,120000.0,Diesel,Individual,Manual,Third Owner,4231,10,Mid,High,0.0,Diesel_Manual,Mahindra,2.92,1
4231,CAR_004232,Toyota Innova 2.5 G (Diesel) 8 Seater BS IV,2011,4461000.0,60000.0,Diesel,Individual,Manual,First Owner,4232,14,High,Very High,0.0,Diesel_Manual,Toyota,3.48,1
4232,CAR_004233,Hyundai Santro Xing XG AT,2004,125000.0,70000.0,Petrol,Individual,Automatic,Second Owner,4233,21,Low,Medium,0.0,Petrol_Automatic,Hyundai,1.79,0
4233,CAR_004234,Tata Indica V2 DLS BSII,2010,4461000.0,100000.0,Diesel,Individual,Manual,Third Owner,4234,15,Low,High,0.0,Diesel_Manual,Tata,0.75,1
4234,CAR_004235,Mahindra KUV 100 G80 K4 Plus,2018,509999.0,15000.0,Petrol,Individual,Manual,First Owner,4235,7,Mid,Low,0.0,Petrol_Manual,Mahindra,34.0,0
4235,CAR_004236,Tata Indigo Classic Dicor,2014,215000.0,60000.0,Diesel,Individual,Manual,Third Owner,4236,11,Low,High,0.0,Diesel_Manual,Tata,2.69,1
4236,CAR_004237,Hyundai Grand i10 1.2 CRDi Asta,2018,465000.0,25000.0,Diesel,Individual,Manual,First Owner,4237,7,Mid,High,0.0,Diesel_Manual,Hyundai,18.6,1
4237,CAR_004238,Maruti Baleno Alpha 1.2,2017,625000.0,60000.0,Petrol,Dealer,Manual,First Owner,4238,8,High,High,0.0,Petrol_Manual,Maruti,12.02,0
4238,CAR_004239,Hyundai Grand i10 CRDi Magna,2017,490000.0,66000.0,Diesel,Dealer,Manual,First Owner,4239,8,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.42,1
4239,CAR_004240,Maruti Ertiga SHVS ZDI,2017,880000.0,64000.0,Diesel,Dealer,Manual,First Owner,4240,8,High,Medium,0.0,Diesel_Manual,Maruti,13.75,1
4240,CAR_004241,Hyundai Santro Xing GL Plus,2013,290000.0,49000.0,Petrol,Individual,Manual,First Owner,4241,12,Low,Medium,0.0,Petrol_Manual,Hyundai,5.92,0
4241,CAR_004242,Tata Sumo GX TC 7 Str BSIII,2006,115999.0,100000.0,Diesel,Individual,Manual,Second Owner,4242,19,Low,High,0.0,Diesel_Manual,Tata,1.16,1
4242,CAR_004243,Maruti Vitara Brezza LDi Option,2017,685000.0,72000.0,Diesel,Dealer,Manual,First Owner,4243,8,High,High,0.0,Diesel_Manual,Maruti,9.51,1
4243,CAR_004244,Honda Jazz S,2009,300000.0,50000.0,Petrol,Individual,Manual,First Owner,4244,16,Low,Medium,0.0,Petrol_Manual,Honda,6.0,0
4244,CAR_004245,Hyundai i20 1.4 Sportz,2017,4461000.0,44000.0,Diesel,Dealer,Manual,First Owner,4245,8,High,Medium,0.0,Diesel_Manual,Hyundai,15.45,1
4245,CAR_004246,Maruti Swift Dzire VDI,2014,480000.0,101000.0,Diesel,Dealer,Manual,First Owner,4246,11,Mid,High,0.0,Diesel_Manual,Maruti,4.75,1
4246,CAR_004247,Maruti Swift VDI,2016,630000.0,55000.0,Diesel,Dealer,Manual,First Owner,4247,9,High,Medium,0.0,Diesel_Manual,Maruti,11.45,1
4247,CAR_004248,Honda City 1.5 GXI,2007,190000.0,115000.0,Petrol,Dealer,Manual,Second Owner,4248,18,Low,High,0.0,Petrol_Manual,Honda,1.65,0
4248,CAR_004249,Maruti Swift VDI,2015,4461000.0,100000.0,Diesel,Dealer,Manual,First Owner,4249,10,Mid,High,0.0,Diesel_Manual,Maruti,5.0,1
4249,CAR_004250,Volkswagen Jetta 2.0L TDI Comfortline,2014,850000.0,52000.0,Diesel,Dealer,Manual,Second Owner,4250,11,High,Medium,0.0,Diesel_Manual,Volkswagen,16.35,1
4250,CAR_004251,Maruti SX4 ZDI,2011,350000.0,105000.0,Diesel,Dealer,Manual,First Owner,4251,14,Mid,High,0.0,Diesel_Manual,Maruti,3.33,1
4251,CAR_004252,Renault Duster 85PS Diesel RxE,2013,425000.0,170000.0,Diesel,Dealer,Manual,First Owner,4252,12,Mid,Very High,0.0,Diesel_Manual,Renault,2.5,1
4252,CAR_004253,Maruti Swift VDI BSIV,2015,495000.0,105000.0,Diesel,Dealer,Manual,First Owner,4253,10,Mid,High,0.0,Diesel_Manual,Maruti,4.71,1
4253,CAR_004254,Maruti Ertiga SHVS VDI,2017,890000.0,60000.0,Diesel,Individual,Manual,Second Owner,4254,8,High,Medium,0.0,Diesel_Manual,Maruti,14.83,1
4254,CAR_004255,Hyundai Verna CRDi 1.6 AT EX,2018,1100000.0,25000.0,Diesel,Individual,Manual,Second Owner,4255,7,Premium,High,0.0,Diesel_Automatic,Hyundai,44.0,1
4255,CAR_004256,Mahindra XUV500 W8 2WD,2014,650000.0,218000.0,Petrol,Individual,Manual,Second Owner,4256,11,High,Very High,0.0,Diesel_Manual,Mahindra,2.98,1
4256,CAR_004257,Hyundai Grand i10 1.2 CRDi Asta,2018,465000.0,60000.0,Diesel,Individual,Manual,First Owner,4257,7,Low,Low,0.0,Diesel_Manual,Hyundai,18.6,1
4257,CAR_004258,Maruti Baleno Alpha 1.2,2017,625000.0,60000.0,Petrol,Dealer,Manual,First Owner,4258,8,High,Medium,0.0,Petrol_Manual,Maruti,12.02,0
4258,CAR_004259,Hyundai Grand i10 CRDi Magna,2017,490000.0,66000.0,Diesel,Dealer,Manual,First Owner,4259,8,Mid,Medium,0.0,Diesel_Manual,Hyundai,7.42,1
4259,CAR_004260,Maruti Ertiga SHVS ZDI,2017,880000.0,60000.0,Diesel,Dealer,Manual,First Owner,4260,8,High,Medium,0.0,Diesel_Manual,Maruti,13.75,1
4260,CAR_004261,Hyundai Santro Xing GL Plus,2013,290000.0,49000.0,Petrol,Individual,Manual,First Owner,4261,12,Low,Medium,0.0,Petrol_Manual,Hyundai,5.92,0
4261,CAR_004262,Tata Sumo GX TC 7 Str BSIII,2006,115999.0,100000.0,CNG,Individual,Manual,Second Owner,4262,19,Low,High,0.0,Diesel_Manual,Tata,1.16,1
4262,CAR_004263,Maruti Vitara Brezza LDi Option,2017,685000.0,72000.0,Diesel,Dealer,Manual,First Owner,4263,8,High,High,0.0,Diesel_Manual,Maruti,9.51,1
4263,CAR_004264,Honda Jazz S,2009,300000.0,50000.0,Petrol,Individual,Manual,First Owner,4264,16,Low,Medium,0.0,Petrol_Manual,Honda,6.0,0
4264,CAR_004265,Hyundai i20 1.4 Sportz,2017,680000.0,60000.0,Diesel,Dealer,Manual,First Owner,4265,8,Low,Medium,0.0,Diesel_Manual,Hyundai,15.45,1
4265,CAR_004266,Maruti Swift Dzire VDI,2014,480000.0,101000.0,Diesel,Dealer,Manual,First Owner,4266,11,Mid,High,0.0,Diesel_Manual,Maruti,4.75,1
4266,CAR_004267,Maruti Swift VDI,2016,630000.0,60000.0,Diesel,Dealer,Manual,First Owner,4267,9,High,Medium,0.0,Diesel_Manual,Maruti,11.45,1
4267,CAR_004268,Honda City 1.5 GXI,2007,190000.0,115000.0,Petrol,Dealer,Manual,Second Owner,4268,18,Low,High,0.0,Petrol_Manual,Honda,1.65,0
4268,CAR_004269,Maruti Swift VDI,2015,500000.0,100000.0,Diesel,Dealer,Manual,First Owner,4269,10,Mid,High,0.0,Diesel_Manual,Maruti,5.0,1
4269,CAR_004270,Volkswagen Jetta 2.0L TDI Comfortline,2014,850000.0,52000.0,Diesel,Dealer,Manual,Second Owner,4270,11,Low,Medium,0.0,Diesel_Manual,Volkswagen,16.35,1
4270,CAR_004271,Maruti SX4 ZDI,2011,350000.0,105000.0,Diesel,Dealer,Manual,First Owner,4271,14,Mid,High,0.0,Diesel_Manual,Maruti,3.33,1
4271,CAR_004272,Renault Duster 85PS Diesel RxE,2013,4461000.0,170000.0,Diesel,Dealer,Manual,First Owner,4272,12,Mid,Very High,0.0,Diesel_Manual,Renault,2.5,1
4272,CAR_004273,Maruti Swift VDI BSIV,2015,495000.0,105000.0,Diesel,Dealer,Manual,First Owner,4273,10,Mid,High,0.0,Diesel_Manual,Maruti,4.71,1
4273,CAR_004274,Maruti Ertiga SHVS VDI,2017,890000.0,60000.0,Diesel,Individual,Manual,Second Owner,4274,8,High,Medium,0.0,Diesel_Manual,Maruti,14.83,1
4274,CAR_004275,Hyundai Verna CRDi 1.6 AT EX,2018,1100000.0,60000.0,Diesel,Individual,Automatic,Second Owner,4275,7,Premium,Low,0.0,Diesel_Automatic,Hyundai,44.0,1
4275,CAR_004276,Mahindra XUV500 W8 2WD,2014,650000.0,218000.0,Diesel,Individual,Manual,Second Owner,4276,11,High,Very High,0.0,Diesel_Manual,Mahindra,2.98,1
4276,CAR_004277,Ford Fiesta 1.6 ZXi Duratec,2009,4461000.0,58000.0,Petrol,Individual,Manual,Third Owner,4277,16,Low,Medium,0.0,Petrol_Manual,Ford,4.02,0
4277,CAR_004278,Tata Indigo CS LS (TDI) BS-III,2015,300000.0,110000.0,Diesel,Individual,Manual,Second Owner,4278,10,Low,High,0.0,Diesel_Manual,Tata,2.73,1
4278,CAR_004279,Honda Amaze S Petrol BSIV,2020,614000.0,1000.0,Petrol,Individual,Manual,First Owner,4279,5,High,Low,0.0,Petrol_Manual,Honda,614.0,0
4279,CAR_004280,Hyundai i10 Sportz 1.2,2010,250000.0,100000.0,Petrol,Individual,Manual,Second Owner,4280,15,Low,High,0.0,Petrol_Manual,Hyundai,2.5,0
4280,CAR_004281,Hyundai i10 Sportz 1.2,2010,250000.0,110000.0,Petrol,Individual,Manual,Second Owner,4281,15,Low,High,0.0,Petrol_Manual,Hyundai,2.27,0
4281,CAR_004282,Maruti Alto LXi,2011,200000.0,40000.0,Petrol,Individual,Manual,First Owner,4282,14,Low,Medium,0.0,Petrol_Manual,Maruti,5.0,0
4282,CAR_004283,Maruti Wagon R LX Minor,2013,290000.0,52000.0,Petrol,Individual,Manual,First Owner,4283,12,Low,High,0.0,Petrol_Manual,Maruti,5.58,0
4283,CAR_004284,Maruti Wagon R LXI BS IV,2013,320000.0,80000.0,Petrol,Individual,Manual,Second Owner,4284,12,Mid,High,0.0,Petrol_Manual,Maruti,4.0,0
4284,CAR_004285,Hyundai Santro Xing XS eRLX Euro III,2006,145000.0,52000.0,Petrol,Individual,Manual,Second Owner,4285,19,Low,Medium,0.0,Petrol_Manual,Hyundai,2.79,0
4285,CAR_004286,Maruti Alto LXI,2005,100000.0,124000.0,Petrol,Individual,Manual,Second Owner,4286,20,Low,High,0.0,Petrol_Manual,Maruti,0.81,0
4286,CAR_004287,Fiat Punto 1.3 Emotion,2010,130000.0,210000.0,Diesel,Individual,Manual,Second Owner,4287,15,Low,Very High,0.0,Diesel_Manual,Fiat,0.62,1
4287,CAR_004288,Hyundai Santro AT,2006,145000.0,66000.0,Petrol,Individual,Automatic,Second Owner,4288,19,Low,Medium,0.0,Petrol_Automatic,Hyundai,2.2,0
4288,CAR_004289,Maruti Alto LX BSIII,2008,4461000.0,120000.0,Petrol,Individual,Manual,Third Owner,4289,17,Low,High,0.0,Petrol_Manual,Maruti,0.92,0
4289,CAR_004290,Maruti Swift Dzire VDI,2019,680000.0,40000.0,Diesel,Individual,Manual,First Owner,4290,6,High,Medium,0.0,Diesel_Manual,Maruti,17.0,1
4290,CAR_004291,Maruti 800 Std,2004,37500.0,90000.0,Petrol,Individual,Manual,Second Owner,4291,21,Low,High,0.0,Petrol_Manual,Maruti,0.42,0
4291,CAR_004292,Maruti Ritz VDi,2011,200000.0,60000.0,Diesel,Individual,Manual,Second Owner,4292,14,Low,High,0.0,Diesel_Manual,Maruti,1.67,1
4292,CAR_004293,Nissan Sunny Diesel XL,2012,300000.0,110000.0,Diesel,Individual,Manual,First Owner,4293,13,Low,High,0.0,Diesel_Manual,Nissan,2.73,1
4293,CAR_004294,Hyundai Santro GS,2005,80000.0,56580.0,Petrol,Dealer,Manual,First Owner,4294,20,Low,High,0.0,Petrol_Manual,Hyundai,1.41,0
4294,CAR_004295,Mahindra XUV500 AT W8 FWD,2015,740000.0,45000.0,LPG,Dealer,Automatic,First Owner,4295,10,High,Medium,0.0,Diesel_Automatic,Mahindra,16.44,1
4295,CAR_004296,Mahindra Scorpio S10 7 Seater,2015,630000.0,50000.0,Electric,Individual,Manual,First Owner,4296,10,High,High,0.0,Diesel_Manual,Mahindra,12.6,1
4296,CAR_004297,Hyundai Santro Xing XG,2005,70000.0,68500.0,Petrol,Dealer,Manual,First Owner,4297,20,Low,Medium,0.0,Petrol_Manual,Hyundai,1.02,0
4297,CAR_004298,Hyundai Santro Sportz AMT,2019,484999.0,5007.0,Petrol,Dealer,Automatic,First Owner,4298,6,Mid,Low,0.0,Petrol_Automatic,Hyundai,96.86,0
4298,CAR_004299,Nissan Micra Active XV S,2013,164000.0,30000.0,Petrol,Individual,Manual,First Owner,4299,12,Low,Low,0.0,Petrol_Manual,Nissan,5.47,0
4299,CAR_004300,Honda City 1.5 V AT,2008,140000.0,70000.0,Petrol,Individual,Automatic,First Owner,4300,17,Low,Medium,0.0,Petrol_Automatic,Honda,2.0,0
4300,CAR_004301,Mercedes-Benz E-Class E250 CDI Elegance,2011,999000.0,49600.0,Diesel,Dealer,Automatic,First Owner,4301,14,High,Medium,0.0,Diesel_Automatic,Mercedes-Benz,20.14,1
4301,CAR_004302,Maruti Alto LXI,2005,56000.0,23000.0,Petrol,Individual,Manual,Second Owner,4302,20,Low,Low,0.0,Petrol_Manual,Maruti,2.43,0
4302,CAR_004303,BMW 7 Series 730Ld,2006,1050000.0,30000.0,Diesel,Dealer,Automatic,First Owner,4303,19,Premium,High,0.0,Diesel_Automatic,BMW,35.0,1
4303,CAR_004304,Hyundai Verna 1.6 VTVT,2010,190000.0,38000.0,Petrol,Dealer,Manual,First Owner,4304,15,Low,Medium,0.0,Petrol_Manual,Hyundai,5.0,0
4304,CAR_004305,Audi Q5 3.0 TDI Quattro Technology,2018,3899000.0,22000.0,Diesel,Dealer,Automatic,First Owner,4305,7,Low,Low,0.0,Diesel_Automatic,Audi,177.23,1
4305,CAR_004306,Maruti Ritz VDi,2011,4461000.0,40000.0,Diesel,Individual,Manual,Second Owner,4306,14,Low,Medium,0.0,Diesel_Manual,Maruti,3.75,1
4306,CAR_004307,Hyundai i10 Sportz 1.2,2011,235000.0,43100.0,Petrol,Dealer,Manual,First Owner,4307,14,Low,Medium,0.0,Petrol_Manual,Hyundai,5.45,0
4307,CAR_004308,Mahindra Xylo H4,2019,599000.0,15000.0,Diesel,Individual,Manual,Third Owner,4308,6,Mid,Low,0.0,Diesel_Manual,Mahindra,39.93,1
4308,CAR_004309,Maruti Alto 800 LXI,2018,4461000.0,35000.0,Petrol,Individual,Manual,First Owner,4309,7,Low,Medium,0.0,Petrol_Manual,Maruti,5.71,0
4309,CAR_004310,Datsun GO Plus T,2017,350000.0,10171.0,Petrol,Dealer,Manual,First Owner,4310,8,Mid,Low,0.0,Petrol_Manual,Datsun,34.41,0
4310,CAR_004311,Renault Duster 110PS Diesel RxL,2015,465000.0,41123.0,Diesel,Dealer,Manual,First Owner,4311,10,Mid,Medium,0.0,Diesel_Manual,Renault,11.31,1
4311,CAR_004312,Toyota Camry Hybrid 2.5,2017,1900000.0,60000.0,Petrol,Dealer,Automatic,First Owner,4312,8,Premium,High,0.0,Petrol_Automatic,Toyota,94.44,0
4312,CAR_004313,Maruti Ertiga 1.5 VDI,2019,1000000.0,15000.0,Diesel,Individual,Manual,First Owner,4313,6,High,Low,0.0,Diesel_Manual,Maruti,66.67,1
4313,CAR_004314,Ford Endeavour 2.2 Titanium AT 4X2,2019,4461000.0,10000.0,Diesel,Individual,Automatic,First Owner,4314,6,Premium,Low,0.0,Diesel_Automatic,Ford,280.0,1
4314,CAR_004315,Maruti Swift Dzire VDI,2015,470000.0,170000.0,Diesel,Individual,Manual,First Owner,4315,10,Mid,Very High,0.0,Diesel_Manual,Maruti,2.76,1
4315,CAR_004316,Maruti Celerio ZXI,2017,415000.0,20000.0,Petrol,Individual,Manual,First Owner,4316,8,Mid,Low,0.0,Petrol_Manual,Maruti,20.75,0
4316,CAR_004317,Nissan Terrano XL 85 PS,2014,500000.0,82000.0,Diesel,Individual,Manual,First Owner,4317,11,Mid,High,0.0,Diesel_Manual,Nissan,6.1,1
4317,CAR_004318,Chevrolet Spark 1.0 LT BS3,2013,150000.0,60000.0,CNG,Individual,Manual,Second Owner,4318,12,Low,Medium,0.0,Petrol_Manual,Chevrolet,2.5,0
4318,CAR_004319,Maruti Alto STD,2005,4461000.0,50000.0,Petrol,Individual,Manual,Second Owner,4319,20,Low,Medium,0.0,Petrol_Manual,Maruti,1.9,0
4319,CAR_004320,Maruti Swift LDI,2012,400000.0,70000.0,Diesel,Individual,Manual,First Owner,4320,13,Mid,Medium,0.0,Diesel_Manual,Maruti,5.71,1
4320,CAR_004321,Maruti Alto LX,2008,114999.0,66782.0,Petrol,Individual,Manual,Second Owner,4321,17,Low,Medium,0.0,Petrol_Manual,Maruti,1.72,0
4321,CAR_004322,Maruti Alto LX,2006,75000.0,130000.0,Petrol,Individual,Manual,First Owner,4322,19,Low,High,0.0,Petrol_Manual,Maruti,0.58,0
4322,CAR_004323,Hyundai Verna 1.6 SX CRDi (O),2013,500000.0,120000.0,Diesel,Individual,Manual,First Owner,4323,12,Mid,High,0.0,Diesel_Manual,Hyundai,4.17,1
4323,CAR_004324,Maruti 800 AC,2014,195000.0,60000.0,Petrol,Individual,Manual,Second Owner,4324,11,Low,High,0.0,Petrol_Manual,Maruti,2.6,0
4324,CAR_004325,Maruti Alto 800 Base,2015,155000.0,40000.0,LPG,Individual,Manual,First Owner,4325,10,Low,Medium,0.0,Petrol_Manual,Maruti,3.88,0
4325,CAR_004326,Maruti Alto LXi,2000,65000.0,90000.0,Petrol,Individual,Manual,Second Owner,4326,25,Low,High,0.0,Petrol_Manual,Maruti,0.72,0
4326,CAR_004327,Honda City 1.5 GXI,2005,65000.0,150000.0,Petrol,Individual,Manual,Third Owner,4327,20,Low,High,0.0,Petrol_Manual,Honda,0.43,0
4327,CAR_004328,Tata Nano XM,2015,190000.0,60000.0,Petrol,Individual,Manual,Second Owner,4328,10,Low,Medium,0.0,Petrol_Manual,Tata,3.17,0
4328,CAR_004329,Mahindra Verito 1.5 D6 BSIII,2012,200000.0,112198.0,Diesel,Individual,Manual,Second Owner,4329,13,Low,High,0.0,Diesel_Manual,Mahindra,1.78,1
4329,CAR_004330,Tata Manza Aura Safire BS IV,2010,160000.0,60000.0,Petrol,Individual,Manual,Second Owner,4330,15,Low,High,0.0,Petrol_Manual,Tata,2.67,0
4330,CAR_004331,Tata Indica Vista Aqua 1.4 TDI,2010,150000.0,130000.0,Diesel,Individual,Manual,Second Owner,4331,15,Low,High,0.0,Diesel_Manual,Tata,1.15,1
4331,CAR_004332,Ford EcoSport 1.5 TDCi Titanium BSIV,2015,530000.0,175000.0,Diesel,Individual,Manual,Third Owner,4332,10,Mid,High,0.0,Diesel_Manual,Ford,3.03,1
4332,CAR_004333,Mahindra Scorpio S2 7 Seater,2015,750000.0,120000.0,Diesel,Individual,Manual,First Owner,4333,10,High,High,0.0,Diesel_Manual,Mahindra,6.25,1
4333,CAR_004334,Maruti Ritz VDi,2012,225000.0,90000.0,Electric,Individual,Manual,Second Owner,4334,13,Low,High,0.0,Diesel_Manual,Maruti,2.5,1
4334,CAR_004335,Toyota Innova 2.5 VX (Diesel) 8 Seater BS IV,2012,600000.0,170000.0,Diesel,Individual,Manual,First Owner,4335,13,Mid,Very High,0.0,Diesel_Manual,Toyota,3.53,1
4335,CAR_004336,Hyundai i20 Magna 1.4 CRDi (Diesel),2014,409999.0,60000.0,Diesel,Individual,Manual,Second Owner,4336,11,Mid,High,0.0,Diesel_Manual,Hyundai,5.12,1
4336,CAR_004337,Hyundai i20 Magna 1.4 CRDi,2014,409999.0,80000.0,Diesel,Individual,Manual,Second Owner,4337,11,Mid,High,0.0,Diesel_Manual,Hyundai,5.12,1
4337,CAR_004338,Maruti 800 AC BSIII,2009,110000.0,83000.0,Petrol,Individual,Manual,Second Owner,4338,16,Low,High,0.0,Petrol_Manual,Maruti,1.33,0
4338,CAR_004339,Hyundai Creta 1.6 CRDi SX Option,2016,865000.0,60000.0,Diesel,Individual,Manual,First Owner,4339,9,High,High,0.0,Diesel_Manual,Hyundai,9.61,1
4339,CAR_004340,Renault KWID RXT,2016,4461000.0,40000.0,Petrol,Individual,Manual,First Owner,4340,9,Low,Medium,0.0,Petrol_Manual,Renault,5.62,0

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e8b842c0-03fb-4c66-be2a-af6f7953522c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc57ca94-5987-49e2-9178-88c47824c0a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Q1. Load the cleaned Car Evaluation dataset. Identify the Type of  ML Problem. Select 'price_category' as target variable, and print the shape of X and y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "646264eb-47db-4a49-8ca0-9109b167f618",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>vehicle_uid</th>\n",
       "      <th>name</th>\n",
       "      <th>year</th>\n",
       "      <th>selling_price</th>\n",
       "      <th>km_driven</th>\n",
       "      <th>fuel</th>\n",
       "      <th>seller_type</th>\n",
       "      <th>transmission</th>\n",
       "      <th>owner</th>\n",
       "      <th>car_record_id</th>\n",
       "      <th>car_age</th>\n",
       "      <th>price_category</th>\n",
       "      <th>km_category</th>\n",
       "      <th>owner_count</th>\n",
       "      <th>fuel_transmission</th>\n",
       "      <th>brand</th>\n",
       "      <th>price_per_km</th>\n",
       "      <th>is_diesel</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>CAR_000001</td>\n",
       "      <td>Maruti 800 AC</td>\n",
       "      <td>2007</td>\n",
       "      <td>60000.0</td>\n",
       "      <td>70000.0</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>1</td>\n",
       "      <td>18</td>\n",
       "      <td>Low</td>\n",
       "      <td>Medium</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Petrol_Manual</td>\n",
       "      <td>Maruti</td>\n",
       "      <td>0.86</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>CAR_000002</td>\n",
       "      <td>Maruti Wagon R LXI Minor</td>\n",
       "      <td>2007</td>\n",
       "      <td>135000.0</td>\n",
       "      <td>50000.0</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>2</td>\n",
       "      <td>18</td>\n",
       "      <td>Low</td>\n",
       "      <td>Medium</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Petrol_Manual</td>\n",
       "      <td>Maruti</td>\n",
       "      <td>2.70</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>CAR_000003</td>\n",
       "      <td>Hyundai Verna 1.6 SX</td>\n",
       "      <td>2012</td>\n",
       "      <td>600000.0</td>\n",
       "      <td>100000.0</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>Low</td>\n",
       "      <td>Medium</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Diesel_Manual</td>\n",
       "      <td>Hyundai</td>\n",
       "      <td>6.00</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>CAR_000004</td>\n",
       "      <td>Datsun RediGO T Option</td>\n",
       "      <td>2017</td>\n",
       "      <td>250000.0</td>\n",
       "      <td>46000.0</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>4</td>\n",
       "      <td>8</td>\n",
       "      <td>Low</td>\n",
       "      <td>Medium</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Petrol_Manual</td>\n",
       "      <td>Datsun</td>\n",
       "      <td>5.43</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>CAR_000005</td>\n",
       "      <td>Honda Amaze VX i-DTEC</td>\n",
       "      <td>2014</td>\n",
       "      <td>450000.0</td>\n",
       "      <td>141000.0</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second Owner</td>\n",
       "      <td>5</td>\n",
       "      <td>11</td>\n",
       "      <td>Low</td>\n",
       "      <td>High</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Diesel_Manual</td>\n",
       "      <td>Honda</td>\n",
       "      <td>3.19</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4335</th>\n",
       "      <td>4335</td>\n",
       "      <td>CAR_004336</td>\n",
       "      <td>Hyundai i20 Magna 1.4 CRDi (Diesel)</td>\n",
       "      <td>2014</td>\n",
       "      <td>409999.0</td>\n",
       "      <td>60000.0</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second Owner</td>\n",
       "      <td>4336</td>\n",
       "      <td>11</td>\n",
       "      <td>Mid</td>\n",
       "      <td>High</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Diesel_Manual</td>\n",
       "      <td>Hyundai</td>\n",
       "      <td>5.12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4336</th>\n",
       "      <td>4336</td>\n",
       "      <td>CAR_004337</td>\n",
       "      <td>Hyundai i20 Magna 1.4 CRDi</td>\n",
       "      <td>2014</td>\n",
       "      <td>409999.0</td>\n",
       "      <td>80000.0</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second Owner</td>\n",
       "      <td>4337</td>\n",
       "      <td>11</td>\n",
       "      <td>Mid</td>\n",
       "      <td>High</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Diesel_Manual</td>\n",
       "      <td>Hyundai</td>\n",
       "      <td>5.12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4337</th>\n",
       "      <td>4337</td>\n",
       "      <td>CAR_004338</td>\n",
       "      <td>Maruti 800 AC BSIII</td>\n",
       "      <td>2009</td>\n",
       "      <td>110000.0</td>\n",
       "      <td>83000.0</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second Owner</td>\n",
       "      <td>4338</td>\n",
       "      <td>16</td>\n",
       "      <td>Low</td>\n",
       "      <td>High</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Petrol_Manual</td>\n",
       "      <td>Maruti</td>\n",
       "      <td>1.33</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4338</th>\n",
       "      <td>4338</td>\n",
       "      <td>CAR_004339</td>\n",
       "      <td>Hyundai Creta 1.6 CRDi SX Option</td>\n",
       "      <td>2016</td>\n",
       "      <td>865000.0</td>\n",
       "      <td>60000.0</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>4339</td>\n",
       "      <td>9</td>\n",
       "      <td>High</td>\n",
       "      <td>High</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Diesel_Manual</td>\n",
       "      <td>Hyundai</td>\n",
       "      <td>9.61</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4339</th>\n",
       "      <td>4339</td>\n",
       "      <td>CAR_004340</td>\n",
       "      <td>Renault KWID RXT</td>\n",
       "      <td>2016</td>\n",
       "      <td>4461000.0</td>\n",
       "      <td>40000.0</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "      <td>4340</td>\n",
       "      <td>9</td>\n",
       "      <td>Low</td>\n",
       "      <td>Medium</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Petrol_Manual</td>\n",
       "      <td>Renault</td>\n",
       "      <td>5.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>4340 rows × 19 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Unnamed: 0 vehicle_uid                                 name  year  \\\n",
       "0              0  CAR_000001                        Maruti 800 AC  2007   \n",
       "1              1  CAR_000002             Maruti Wagon R LXI Minor  2007   \n",
       "2              2  CAR_000003                 Hyundai Verna 1.6 SX  2012   \n",
       "3              3  CAR_000004               Datsun RediGO T Option  2017   \n",
       "4              4  CAR_000005                Honda Amaze VX i-DTEC  2014   \n",
       "...          ...         ...                                  ...   ...   \n",
       "4335        4335  CAR_004336  Hyundai i20 Magna 1.4 CRDi (Diesel)  2014   \n",
       "4336        4336  CAR_004337           Hyundai i20 Magna 1.4 CRDi  2014   \n",
       "4337        4337  CAR_004338                  Maruti 800 AC BSIII  2009   \n",
       "4338        4338  CAR_004339     Hyundai Creta 1.6 CRDi SX Option  2016   \n",
       "4339        4339  CAR_004340                     Renault KWID RXT  2016   \n",
       "\n",
       "      selling_price  km_driven    fuel seller_type transmission         owner  \\\n",
       "0           60000.0    70000.0  Petrol  Individual       Manual   First Owner   \n",
       "1          135000.0    50000.0  Petrol  Individual       Manual   First Owner   \n",
       "2          600000.0   100000.0  Diesel  Individual       Manual   First Owner   \n",
       "3          250000.0    46000.0  Petrol  Individual       Manual   First Owner   \n",
       "4          450000.0   141000.0  Diesel  Individual       Manual  Second Owner   \n",
       "...             ...        ...     ...         ...          ...           ...   \n",
       "4335       409999.0    60000.0  Diesel  Individual       Manual  Second Owner   \n",
       "4336       409999.0    80000.0  Diesel  Individual       Manual  Second Owner   \n",
       "4337       110000.0    83000.0  Petrol  Individual       Manual  Second Owner   \n",
       "4338       865000.0    60000.0  Diesel  Individual       Manual   First Owner   \n",
       "4339      4461000.0    40000.0  Petrol  Individual       Manual   First Owner   \n",
       "\n",
       "      car_record_id  car_age price_category km_category  owner_count  \\\n",
       "0                 1       18            Low      Medium          0.0   \n",
       "1                 2       18            Low      Medium          0.0   \n",
       "2                 3       13            Low      Medium          0.0   \n",
       "3                 4        8            Low      Medium          0.0   \n",
       "4                 5       11            Low        High          0.0   \n",
       "...             ...      ...            ...         ...          ...   \n",
       "4335           4336       11            Mid        High          0.0   \n",
       "4336           4337       11            Mid        High          0.0   \n",
       "4337           4338       16            Low        High          0.0   \n",
       "4338           4339        9           High        High          0.0   \n",
       "4339           4340        9            Low      Medium          0.0   \n",
       "\n",
       "     fuel_transmission    brand  price_per_km  is_diesel  \n",
       "0        Petrol_Manual   Maruti          0.86          0  \n",
       "1        Petrol_Manual   Maruti          2.70          0  \n",
       "2        Diesel_Manual  Hyundai          6.00          1  \n",
       "3        Petrol_Manual   Datsun          5.43          0  \n",
       "4        Diesel_Manual    Honda          3.19          1  \n",
       "...                ...      ...           ...        ...  \n",
       "4335     Diesel_Manual  Hyundai          5.12          1  \n",
       "4336     Diesel_Manual  Hyundai          5.12          1  \n",
       "4337     Petrol_Manual   Maruti          1.33          0  \n",
       "4338     Diesel_Manual  Hyundai          9.61          1  \n",
       "4339     Petrol_Manual  Renault          5.62          0  \n",
       "\n",
       "[4340 rows x 19 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(r\"E:\\pandas\\Machine Learning\\Projects\\Mock Test data\\Car Evalution.csv\")\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "275e5320-b40e-4389-a54e-f8bc523dd780",
   "metadata": {},
   "outputs": [],
   "source": [
    "# target and feature data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "376a8807-7be7-410c-b721-15b961223e26",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = data.drop(\"price_category\",axis=1)\n",
    "y = data[\"price_category\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f232fa09-aa3e-4be1-951b-018a0477f0fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# shape of X and Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a9c5111e-1c49-4765-87e2-d1e7a3949dca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4340, 18)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ef407ca0-45c3-47ec-9cda-21e92222c1e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4340,)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a6be9149-a178-403e-ab32-c8566dcbb4e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q2. Identify categorical columns, apply Encoding techniques (if necessary), perform Standard Scaling on numerical features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8ce94c6b-75b9-4586-b4d8-4d206b6f881b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder, StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0806e207-887a-44bc-bd30-31b7aa5168fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "le = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "05db287a-43ee-4717-827a-4b6205ad96c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in x.select_dtypes(include=\"object\").columns:\n",
    "    x[col] = le.fit_transform(x[col])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1f0a1fa7-e899-4249-8697-689388e536a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c2cd7d8d-dfe7-42da-8820-e7dc1eb4c5d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-1.73165176, -1.73165176,  0.05852176, ...,  0.35817253,\n",
       "        -0.02292873, -0.99219635],\n",
       "       [-1.73085358, -1.73085358,  0.72352031, ...,  0.35817253,\n",
       "        -0.02244466, -0.99219635],\n",
       "       [-1.7300554 , -1.7300554 , -0.61647677, ..., -0.81667398,\n",
       "        -0.02157648,  1.00786503],\n",
       "       ...,\n",
       "       [ 1.7300554 ,  1.7300554 ,  0.06352175, ...,  0.35817253,\n",
       "        -0.02280508, -0.99219635],\n",
       "       [ 1.73085358,  1.73085358, -0.92647609, ..., -0.81667398,\n",
       "        -0.02062675,  1.00786503],\n",
       "       [ 1.73165176,  1.73165176,  1.0060197 , ...,  1.0924516 ,\n",
       "        -0.02167646, -0.99219635]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_scaled = scaler.fit_transform(x)\n",
    "x_scaled "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "78137513-cd41-442d-91a0-54907e2e64ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q.3.Plot the distribution of car price categories. Analyse the relationship  between safety, selling price indicators, and number of owners or usage (km driven) with the target variable price category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "40e7bbe8-a7a5-4844-8276-7de94c231695",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "44662ed0-13d8-4b32-ac14-23a3de944f6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHECAYAAADYuDUfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABH40lEQVR4nO3de3zP9f//8fvbxgzb2w52qhmLZhpyivl8cshxOSQVokk0Ph+HfhqRTyf1EemkDz75lpwp+pTUp7SQrOQYJoRGhGzmMJvD2obn74++e329bUOMbV636+Xyuly8nq/n6/l6vN5v2r3n6/l+z2GMMQIAALCxMsVdAAAAQHEjEAEAANsjEAEAANsjEAEAANsjEAEAANsjEAEAANsjEAEAANsjEAEAANsjEAEAANsjEAEoVNOmTZWcnFzcZRS51157TdWrV5ePj49+/fXXAvt069ZNXl5eqlixolq1aqUNGzbc4CoB3EgEIqAEeOqpp3Tbbbfptdde07FjxwrtN3HiRDVq1OiKx+3UqZO+/PLLK+5/8fjr1q2Tu7v7FZ9/sdOnTys8PFxHjx695HVupISEBI0cOVKtW7fW/PnzFRoaWmi/YcOGafLkyXJzc9Nf//pXffvtt5cd/6OPPlJYWJiys7OLunRLSkqKevbsKS8vL1WuXFlVqlRRZmbmFZ1br149bd269brVBpRaBkCxa9GihQkPDzchISHGw8PD9OnTx+zcuTNfv7S0NLNp06YrHjcsLMyMHTv2ivtfPL4ks3fv3is+/2J79+41ksyqVasueZ0b6W9/+5tp2rTpZft5eHiYb775xhhjzPnz502nTp1MixYtLnteZmamWb169TVWWbhffvnF3HLLLSYyMtJMmzbNPP/888bhcJhffvnlis6XZObNm3fd6gNKK2aIgBLi7rvv1q+//qr58+dr586datCggaZPn+7Sp0qVKqpfv75LW1JSksaMGaO6devqL3/5S75xz549e8U1FDR+Ubi4hut1nSuxY8cONW/e/JJ9zp8/r5ycHHl6ekqSHA6H2rZtq3379l12fC8vL0VHRxdFqflkZ2era9euCg4O1rp16/T444/rxRdf1NGjR1W9evUrHufP/J0A7IJABJQg7u7ueuCBB7R27VqNHTtWAwYM0Lx586zjvXv31qRJkyRJxhj17NlTzZo1U3Jystq3b69Nmzbp/PnzatOmjSpXrqyDBw/q1Vdflbu7u9q2bWuN8cYbb+ill15S9erV5eXlpe+//z7f+Hnmz5+vunXrytPTU61atdJPP/1kHbvzzju1ePFia3/ChAnq3bu3zpw5Ix8fHyv0dOrUSe7u7ho7dmyB10lOTla7du1UoUIFVatWzeUxX/fu3TV9+nSNHTtWvr6+qlu3rvbv31/oa/jll1+qbt26Kl++vJo0aaJffvlFkvTKK6/I3d1diYmJevXVV+VwOHTLLbcoNzc33xgnT56UMUaVKlWy2r755hvVqFFDkvSPf/xD8fHx+ve//63IyEhVqFBBCxcutI4NHz7cZbx3331XgYGB8vHxUY8ePXTw4EGX1+yWW25R+fLldeedd2rZsmWF3tuECRN04MABLVq0SF5eXla7r6+v9ecZM2aoVq1a8vDwUP369a3xatasKT8/P0nS4MGD5ebmpri4OEl/BKQRI0bI399fFSpUUHR0tDZv3myNuX37dnXp0kWVK1eWn5+fnn32Wf3++++SpFOnTikuLk6+vr7y8fHR+PHjrfMKe53++te/uvy9znvfIiMjC7134Lor7ikqAH88Mnv00UfztY8aNcr4+fmZ7OxsY4wx9evXN6NGjTLGGPPdd98ZSearr76y+h8/ftwYY8yePXvM559/bsLDw02HDh3Mhx9+aJKSkqxrlSlTxjRu3Nh88sknpm7dutZjtQvHN+aPxyuBgYHm5ZdfNu+9955p0qSJCQ0NNadPnzbZ2dnG4XCYzz77zOr/yCOPmA4dOhhjjPnhhx/MggULjCTz5JNPmv/85z/mt99+y3edrKwsExYWZqKjo827775r7r33XuN0Os2pU6eMMcY0adLEBAcHm8DAQDNp0iQTHh7uUuOFtmzZYsqWLWseeeQRM23aNBMREWHatGljjDFm//795qOPPjJVqlQxAwYMMP/5z3+sR2IXy3vUt3btWvPTTz+ZQYMGGYfDYZYsWWKMMebRRx81ZcqUMbfffruZP3++6dChg3n88ceNMcbcf//9pkePHtZYn376qXE4HObpp582M2bMMG3btjULFy40xhgzbdo0U758efPcc8+ZWbNmmfvuu89ERkYWWFN2drapXLmyef311ws8bowxr7zyipFkevXqZWbMmGHq1atn1bV9+3bz2WefmUqVKplHHnnEfPjhhyY5OdkYY8wzzzxjKleubCZMmGBmzJhh7r77bhMTE2OMMWb37t3G6XSaxo0bm0mTJpmpU6eagIAA89xzzxljjOnRo4cJCQkxkyZNMk8++aTLI9LCXqeBAweaypUrmyNHjhhj/ngkGRUVZV0TKA4EIqAEKCwQbdmyxUgyGzZsMMYYU6NGDTN+/HhjjDHbtm0zbm5uRpKJiIgw77zzTr7zmzVrZoYOHZrvWrVr1zZnzpwxxhgTHx9vvvjii3zjG/NHIPrpp5+s/RMnThin02nmzJljDh48aCSZbdu2Wce7detmunfvbu3n5uYaSebjjz92qeHC67z33nvG19fXnDx50hhjzLFjx4ybm5v5/PPPjTHG1KtXz3h4eFjXeeSRR0zv3r0LfB0feeQR07ZtW2s/L4zkjW3MH+uqpk6dWuD5eX788UcjydpCQ0PNRx99ZB1/9NFHTZUqVawf6K+//rqZNm2aMcaYNm3amIEDBxpj/vhBHxERYQYMGFDgdcLDw82bb75pjh49av75z38af39/8/LLLxfYd+XKlcbhcJijR48WeDwjI8OUL1/ejBkzxmpr06ZNvvAYEhJi3njjDWv/zJkzpmLFimbRokVm//79ZsSIEdZ7bIwx/fr1M/Xr1ze5ubnWOTVr1jSvvvqqSU5ONg6Hw3z//ffWsfr165sRI0Zc8nU6dOiQ8fT0NEOGDDHGGPPf//7XSCo0oAI3wtV/fATAdZf3ibO8RzdnzpxRhQoVJEl33HGH1qxZo48++khLly7VwIEDlZOToyFDhljnV6hQQWfOnMk3bufOna31MW+88YbVfuH4efL6SZLT6VTNmjX1yy+/KCYmRpLrepRz58659Hd3d5e7u3u+Gi68zpYtW9SsWTPrHn19feXv72/d+7FjxxQbG6s77rjDGjMrK6vA12vLli169NFHrf3bb79dxhgdP37c5fHX5eTVu27dOvn4+Cg8PFxubm4ufe655x75+/tLkssjsgvvbffu3dq1a5c+/fTTfNc4efKkfvnlFyUmJmrMmDHq2rWrvv76a9WtW7fAmlJTU+Xt7W099rrY9u3b9fvvv2vgwIFWm4eHR75+F/+d2L17t37//XfNmTNHjz32mPr06aNNmzYpPDzceg169+5tfdowOztbe/fuVUREhH788UdVrFhRzZo1s8a7/fbbXT4pWdjrNHjwYE2aNEmjRo3SW2+9pcaNG6tly5YF3htwI7CGCCihzp49q5dfflmRkZGqVauWpD/WDTkcDqtP48aNNWHCBG3evFk9evTQJ5984jJGxYoVCwxEhbl4/IulpaVp586dqlatmnx9fVWuXDmlpaVdcsyCarjwOpUrV9ahQ4esY0ePHtWxY8esH8hHjx5Vq1atXMY7fvx4gde6eKxdu3bJw8NDISEhl6zxYnkh76677lLNmjXzhaFLufDeClqflCcnJ0fSHwFw586dmj17turWravDhw+73EOe8PBwZWRkaPv27VZbbm6unnnmGb355pvy8fGR9EfQyhMcHJzvtbr4/cjNzdW5c+cUGRmpffv2adKkSQoPD9e+ffuUnp4uDw8Pl/uYM2eOzp49q6CgIFWuXFlZWVlKT0+37v3nn3+23rtLefrpp+Xh4aG//e1v+vrrrzVy5MjLngNcT8wQASVEYmKiJk2aJH9/fx06dEjvv/++fv75Z5dFtuXKlbN+kLZu3VqBgYHq0KGDzpw5o82bN+f79JQx5k99oujC8fOMHDlSHTp00O+//67JkycrJCRE3bt3V5kyZRQREaGpU6fq6NGjOnz4sLZu3arGjRtftoYLr/Pggw/qlVdeUVxcnJo1a6Z///vfqlOnjvWJOWOMy6zVLbfcokWLFhVY/8MPP6wnn3xSfn5+qlSpksaNG6e///3vLt+ldOrUKZUtW/aSr4Mx5jKvVOEuvLdatWqpQYMG6tq1q4YOHSqHw6GFCxeqY8eOeuqpp9S6dWstWbJE4eHhCgsL0+7duzV9+nT17dvXWoCep3HjxmrevLk6duyoUaNGKTs7W9OmTdOePXs0Z84c1ahRQ2FhYerfv7/69++vX3/9VYsXL1alSpVcQtrF70e9evV0++236z//+Y8qV66swMBA/fTTT3r33Xf12muvqVOnTpoyZYr193LmzJmSJB8fH0VFRSk0NFRdu3ZVv3799OWXX2rXrl3q16/fZV8nPz8/Pfnkk3rppZdUo0YNdevW7apfc6BIFM+TOgAXio2NtdareHh4mKpVq5rY2FiX9TnGGNO2bVvz/vvvG2P+WB/TuHFjU758eVO5cmXz0EMPmWPHjrn0f/zxx63Fr3kefvhh869//avAOi4c3xhjIiIiTNu2bY2Xl5epVKmSeeihh6yF0cYYs2zZMlOzZk1Tvnx5U79+fdO+fXszaNAglzEjIiLM0qVLL3mdRYsWmYiICFO+fHnTtm1bs2/fPutYs2bNrPVExhizfPlyc9tttxVY//nz582YMWNMUFCQ8fb2NgMGDDC///67S59y5cqZTz/9tMDz8+zatavQaxhjzNNPP23i4+MLPNa/f3/zyiuvWPspKSnmkUceMVWqVDG+vr6mf//+5uDBg8aYPxbBDxkyxISGhhoPDw8TGRlpXnzxRZOVlVXg2IcPHzYPPPCA8fT0NL6+vqZnz55m+/bt1vEffvjBNGvWzFSqVMncddddZsaMGaZmzZour2fbtm3Ne++95zLu/v37TZ8+fUxQUJD1Xv7rX/8y586dM1lZWWbQoEHGz8/PhIeHmw8//NBIstYFbd261dx9993G09PT3HHHHWbFihVX9DoZY8zOnTuNpMuu6QJuBIcx1/C/QgBQyqxZs0YNGzZUuXLliruUUmnNmjW6++67lZ2d/aceJRbk+eef19SpU7V//36XtWdAceCRGQBbuV5fmmgXS5YsUe3ata85DJ09e1bTpk1T//79CUMoEQhEAIBCrV+/3voizJ9++kmvvvqqxo0bd83jLl26VKmpqXrssceueSygKBCIAACFmj17tubNm6czZ87o1ltv1dNPP60nn3zymsc9duyYWrRooYiIiCKoErh2rCECAAC2x/cQAQAA2yMQAQAA22MN0RU6f/68Dh06JC8vr0t+ky8AACg5jDE6efKkQkJCVKZM4fNABKIrdOjQIYWGhhZ3GQAA4CocOHBAt956a6HHCURXyMvLS9IfL6i3t3cxVwMAAK5EZmamQkNDrZ/jhSEQXaG8x2Te3t4EIgAASpnLLXdhUTUAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA99+Iu4GbW8Kk5xV0C/tfG1/oUdwkAgBKMGSIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7xRqIxo8fr8aNG8vLy0sBAQHq2rWrdu3a5dLHGKMxY8YoJCREnp6eatmypbZv3+7SJzs7W0OHDpW/v78qVqyoLl266ODBgy590tPTFRsbK6fTKafTqdjYWJ04ceJ63yIAACgFijUQJSYmavDgwVq7dq2WLVums2fPql27djp9+rTV59VXX9Wbb76pKVOmaMOGDQoKClLbtm118uRJq8+wYcP0ySefaMGCBVq1apVOnTqlTp066dy5c1afXr16KSkpSQkJCUpISFBSUpJiY2Nv6P0CAICSyWGMMcVdRJ4jR44oICBAiYmJat68uYwxCgkJ0bBhwzRq1ChJf8wGBQYGasKECRo4cKAyMjJUpUoVzZ07Vz169JAkHTp0SKGhoVqyZInat2+vHTt2qHbt2lq7dq2aNGkiSVq7dq2io6O1c+dORUREXLa2zMxMOZ1OZWRkyNvb+4rup+FTc67ylUBR2/han+IuAQBQDK7053eJWkOUkZEhSfL19ZUk7d27V6mpqWrXrp3Vx8PDQy1atNDq1aslSRs3blRubq5Ln5CQEEVFRVl91qxZI6fTaYUhSWratKmcTqfVBwAA2Jd7cReQxxij+Ph4/fWvf1VUVJQkKTU1VZIUGBjo0jcwMFC//vqr1adcuXLy8fHJ1yfv/NTUVAUEBOS7ZkBAgNXnYtnZ2crOzrb2MzMzr/LOAABASVdiZoiGDBmiH3/8UR988EG+Yw6Hw2XfGJOv7WIX9ymo/6XGGT9+vLUA2+l0KjQ09EpuAwAAlEIlIhANHTpUn332mb755hvdeuutVntQUJAk5ZvFSUtLs2aNgoKClJOTo/T09Ev2OXz4cL7rHjlyJN/sU57Ro0crIyPD2g4cOHD1NwgAAEq0Yg1ExhgNGTJEixYt0ooVK1S9enWX49WrV1dQUJCWLVtmteXk5CgxMVHNmjWTJDVs2FBly5Z16ZOSkqJt27ZZfaKjo5WRkaH169dbfdatW6eMjAyrz8U8PDzk7e3tsgEAgJtTsa4hGjx4sN5//319+umn8vLysmaCnE6nPD095XA4NGzYMI0bN041a9ZUzZo1NW7cOFWoUEG9evWy+vbv31/Dhw+Xn5+ffH19NWLECNWpU0dt2rSRJEVGRqpDhw6Ki4vTO++8I0kaMGCAOnXqdEWfMAMAADe3Yg1EU6dOlSS1bNnSpX3mzJnq27evJGnkyJHKysrSoEGDlJ6eriZNmmjp0qXy8vKy+k+cOFHu7u7q3r27srKy1Lp1a82aNUtubm5Wn/nz5+uJJ56wPo3WpUsXTZky5freIAAAKBVK1PcQlWR8D1HpxvcQAYA9lcrvIQIAACgOBCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7BCIAAGB7xRqIvv32W3Xu3FkhISFyOBxavHixy3GHw1Hg9tprr1l9WrZsme94z549XcZJT09XbGysnE6nnE6nYmNjdeLEiRtwhwAAoDQo1kB0+vRp1atXT1OmTCnweEpKiss2Y8YMORwOPfDAAy794uLiXPq98847Lsd79eqlpKQkJSQkKCEhQUlJSYqNjb1u9wUAAEoX9+K8eExMjGJiYgo9HhQU5LL/6aefqlWrVgoPD3dpr1ChQr6+eXbs2KGEhAStXbtWTZo0kSRNmzZN0dHR2rVrlyIiIq7xLgAAQGlXatYQHT58WF988YX69++f79j8+fPl7++vO+64QyNGjNDJkyetY2vWrJHT6bTCkCQ1bdpUTqdTq1evLvR62dnZyszMdNkAAMDNqVhniP6M2bNny8vLS926dXNp7927t6pXr66goCBt27ZNo0eP1pYtW7Rs2TJJUmpqqgICAvKNFxAQoNTU1EKvN378eL344otFexMAAKBEKjWBaMaMGerdu7fKly/v0h4XF2f9OSoqSjVr1lSjRo20adMmNWjQQNIfi7MvZowpsD3P6NGjFR8fb+1nZmYqNDT0Wm8DAACUQKUiEH333XfatWuXFi5ceNm+DRo0UNmyZZWcnKwGDRooKChIhw8fztfvyJEjCgwMLHQcDw8PeXh4XFPdAACgdCgVa4imT5+uhg0bql69epftu337duXm5io4OFiSFB0drYyMDK1fv97qs27dOmVkZKhZs2bXrWYAAFB6FOsM0alTp7R7925rf+/evUpKSpKvr6+qVq0q6Y9HVf/5z3/0xhtv5Dt/z549mj9/vu699175+/vrp59+0vDhw1W/fn395S9/kSRFRkaqQ4cOiouLsz6OP2DAAHXq1IlPmAEAAEnFPEP0ww8/qH79+qpfv74kKT4+XvXr19fzzz9v9VmwYIGMMXr44YfznV+uXDl9/fXXat++vSIiIvTEE0+oXbt2Wr58udzc3Kx+8+fPV506ddSuXTu1a9dOdevW1dy5c6//DQIAgFLBYYwxxV1EaZCZmSmn06mMjAx5e3tf0TkNn5pznavCldr4Wp/iLgEAUAyu9Od3qVhDBAAAcD0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0VayD69ttv1blzZ4WEhMjhcGjx4sUux/v27SuHw+GyNW3a1KVPdna2hg4dKn9/f1WsWFFdunTRwYMHXfqkp6crNjZWTqdTTqdTsbGxOnHixHW+OwAAUFoUayA6ffq06tWrpylTphTap0OHDkpJSbG2JUuWuBwfNmyYPvnkEy1YsECrVq3SqVOn1KlTJ507d87q06tXLyUlJSkhIUEJCQlKSkpSbGzsdbsvAABQurgX58VjYmIUExNzyT4eHh4KCgoq8FhGRoamT5+uuXPnqk2bNpKkefPmKTQ0VMuXL1f79u21Y8cOJSQkaO3atWrSpIkkadq0aYqOjtauXbsUERFRtDcFAABKnRK/hmjlypUKCAjQ7bffrri4OKWlpVnHNm7cqNzcXLVr185qCwkJUVRUlFavXi1JWrNmjZxOpxWGJKlp06ZyOp1Wn4JkZ2crMzPTZQMAADenEh2IYmJiNH/+fK1YsUJvvPGGNmzYoHvuuUfZ2dmSpNTUVJUrV04+Pj4u5wUGBio1NdXqExAQkG/sgIAAq09Bxo8fb605cjqdCg0NLcI7AwAAJUmxPjK7nB49elh/joqKUqNGjRQWFqYvvvhC3bp1K/Q8Y4wcDoe1f+GfC+tzsdGjRys+Pt7az8zMJBQBAHCTKtEzRBcLDg5WWFiYkpOTJUlBQUHKyclRenq6S7+0tDQFBgZafQ4fPpxvrCNHjlh9CuLh4SFvb2+XDQAA3JxKVSA6duyYDhw4oODgYElSw4YNVbZsWS1btszqk5KSom3btqlZs2aSpOjoaGVkZGj9+vVWn3Xr1ikjI8PqAwAA7K1YH5mdOnVKu3fvtvb37t2rpKQk+fr6ytfXV2PGjNEDDzyg4OBg7du3T//4xz/k7++v+++/X5LkdDrVv39/DR8+XH5+fvL19dWIESNUp04d61NnkZGR6tChg+Li4vTOO+9IkgYMGKBOnTrxCTMAACCpmAPRDz/8oFatWln7eWt2Hn30UU2dOlVbt27VnDlzdOLECQUHB6tVq1ZauHChvLy8rHMmTpwod3d3de/eXVlZWWrdurVmzZolNzc3q8/8+fP1xBNPWJ9G69KlyyW/+wgAANiLwxhjiruI0iAzM1NOp1MZGRlXvJ6o4VNzrnNVuFIbX+tT3CUAAIrBlf78LlVriAAAAK4HAhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALA9AhEAALC9Yg1E3377rTp37qyQkBA5HA4tXrzYOpabm6tRo0apTp06qlixokJCQtSnTx8dOnTIZYyWLVvK4XC4bD179nTpk56ertjYWDmdTjmdTsXGxurEiRM34A4BAEBpUKyB6PTp06pXr56mTJmS79iZM2e0adMmPffcc9q0aZMWLVqkn3/+WV26dMnXNy4uTikpKdb2zjvvuBzv1auXkpKSlJCQoISEBCUlJSk2Nva63RcAAChd3Ivz4jExMYqJiSnwmNPp1LJly1zaJk+erLvuukv79+9X1apVrfYKFSooKCiowHF27NihhIQErV27Vk2aNJEkTZs2TdHR0dq1a5ciIiKK6G4AAEBpVayB6M/KyMiQw+FQ5cqVXdrnz5+vefPmKTAwUDExMXrhhRfk5eUlSVqzZo2cTqcVhiSpadOmcjqdWr16daGBKDs7W9nZ2dZ+ZmZm0d8QbioNn5pT3CXgf218rU9xlwCglCk1gej333/X008/rV69esnb29tq7927t6pXr66goCBt27ZNo0eP1pYtW6zZpdTUVAUEBOQbLyAgQKmpqYVeb/z48XrxxReL/kYAAECJUyoCUW5urnr27Knz58/r7bffdjkWFxdn/TkqKko1a9ZUo0aNtGnTJjVo0ECS5HA48o1pjCmwPc/o0aMVHx9v7WdmZio0NPRabwUAAJRAJT4Q5ebmqnv37tq7d69WrFjhMjtUkAYNGqhs2bJKTk5WgwYNFBQUpMOHD+frd+TIEQUGBhY6joeHhzw8PK65fgAAUPKV6O8hygtDycnJWr58ufz8/C57zvbt25Wbm6vg4GBJUnR0tDIyMrR+/Xqrz7p165SRkaFmzZpdt9oBAEDpUawzRKdOndLu3but/b179yopKUm+vr4KCQnRgw8+qE2bNunzzz/XuXPnrDU/vr6+KleunPbs2aP58+fr3nvvlb+/v3766ScNHz5c9evX11/+8hdJUmRkpDp06KC4uDjr4/gDBgxQp06d+IQZAACQdJUzRPfcc0+BX2yYmZmpe+6554rH+eGHH1S/fn3Vr19fkhQfH6/69evr+eef18GDB/XZZ5/p4MGDuvPOOxUcHGxtq1evliSVK1dOX3/9tdq3b6+IiAg98cQTateunZYvXy43NzfrOvPnz1edOnXUrl07tWvXTnXr1tXcuXOv5tYBAMBN6KpmiFauXKmcnJx87b///ru+++67Kx6nZcuWMsYUevxSxyQpNDRUiYmJl72Or6+v5s2bd8V1AQAAe/lTgejHH3+0/vzTTz+5fGz93LlzSkhI0C233FJ01QEAANwAfyoQ3XnnndbvCyvo0Zinp6cmT55cZMUBAADcCH8qEO3du1fGGIWHh2v9+vWqUqWKdaxcuXIKCAhwWbsDAABQGvypQBQWFiZJOn/+/HUpBgAAoDhc9cfuf/75Z61cuVJpaWn5AtLzzz9/zYUBAADcKFcViKZNm6a///3v8vf3V1BQkMuvwHA4HAQiAABQqlxVIBo7dqxefvlljRo1qqjrAQAAuOGu6osZ09PT9dBDDxV1LQAAAMXiqgLRQw89pKVLlxZ1LQAAAMXiqh6Z1ahRQ88995zWrl2rOnXqqGzZsi7Hn3jiiSIpDgAA4Ea4qkD07rvvqlKlSkpMTMz3qzMcDgeBCAAAlCpXFYj27t1b1HUAAAAUm6taQwQAAHAzuaoZon79+l3y+IwZM66qGAAAgOJwVYEoPT3dZT83N1fbtm3TiRMnCvylrwAAACXZVQWiTz75JF/b+fPnNWjQIIWHh19zUQAAADdSka0hKlOmjJ588klNnDixqIYEAAC4IYp0UfWePXt09uzZohwSAADguruqR2bx8fEu+8YYpaSk6IsvvtCjjz5aJIUBAADcKFcViDZv3uyyX6ZMGVWpUkVvvPHGZT+BBgAAUNJcVSD65ptviroOAACAYnNVgSjPkSNHtGvXLjkcDt1+++2qUqVKUdUFAABww1zVourTp0+rX79+Cg4OVvPmzXX33XcrJCRE/fv315kzZ4q6RgAAgOvqqgJRfHy8EhMT9d///lcnTpzQiRMn9OmnnyoxMVHDhw8v6hoBAACuq6t6ZPbxxx/ro48+UsuWLa22e++9V56enurevbumTp1aVPUBAABcd1c1Q3TmzBkFBgbmaw8ICOCRGQAAKHWuKhBFR0frhRde0O+//261ZWVl6cUXX1R0dHSRFQcAAHAjXNUjs7feeksxMTG69dZbVa9ePTkcDiUlJcnDw0NLly4t6hoBAACuq6sKRHXq1FFycrLmzZunnTt3yhijnj17qnfv3vL09CzqGgEAAK6rqwpE48ePV2BgoOLi4lzaZ8yYoSNHjmjUqFFFUhwAAMCNcFVriN555x3VqlUrX/sdd9yh//mf/7nicb799lt17txZISEhcjgcWrx4sctxY4zGjBmjkJAQeXp6qmXLltq+fbtLn+zsbA0dOlT+/v6qWLGiunTpooMHD7r0SU9PV2xsrJxOp5xOp2JjY3XixIkrrhMAANzcrioQpaamKjg4OF97lSpVlJKScsXjnD59WvXq1dOUKVMKPP7qq6/qzTff1JQpU7RhwwYFBQWpbdu2OnnypNVn2LBh+uSTT7RgwQKtWrVKp06dUqdOnXTu3DmrT69evZSUlKSEhAQlJCQoKSlJsbGxf+KOAQDAzeyqHpmFhobq+++/V/Xq1V3av//+e4WEhFzxODExMYqJiSnwmDFGb731lp555hl169ZNkjR79mwFBgbq/fff18CBA5WRkaHp06dr7ty5atOmjSRp3rx5Cg0N1fLly9W+fXvt2LFDCQkJWrt2rZo0aSJJmjZtmqKjo7Vr1y5FRERczUsAAABuIlc1Q/T4449r2LBhmjlzpn799Vf9+uuvmjFjhp588sl864qu1t69e5Wamqp27dpZbR4eHmrRooVWr14tSdq4caNyc3Nd+oSEhCgqKsrqs2bNGjmdTisMSVLTpk3ldDqtPgAAwN6uaoZo5MiROn78uAYNGqScnBxJUvny5TVq1CiNHj26SApLTU2VpHxfABkYGKhff/3V6lOuXDn5+Pjk65N3fmpqqgICAvKNHxAQYPUpSHZ2trKzs639zMzMq7sRAABQ4l3VDJHD4dCECRN05MgRrV27Vlu2bNHx48f1/PPPF3V9cjgcLvvGmHxtF7u4T0H9LzfO+PHjrUXYTqdToaGhf7JyAABQWlxVIMpTqVIlNW7cWFFRUfLw8CiqmiRJQUFBkpRvFictLc2aNQoKClJOTo7S09Mv2efw4cP5xj9y5EiBv34kz+jRo5WRkWFtBw4cuKb7AQAAJdc1BaLrqXr16goKCtKyZcustpycHCUmJqpZs2aSpIYNG6ps2bIufVJSUrRt2zarT3R0tDIyMrR+/Xqrz7p165SRkWH1KYiHh4e8vb1dNgAAcHO6qjVEReXUqVPavXu3tb93714lJSXJ19dXVatW1bBhwzRu3DjVrFlTNWvW1Lhx41ShQgX16tVLkuR0OtW/f38NHz5cfn5+8vX11YgRI1SnTh3rU2eRkZHq0KGD4uLi9M4770iSBgwYoE6dOvEJMwAAIKmYA9EPP/ygVq1aWfvx8fGSpEcffVSzZs3SyJEjlZWVpUGDBik9PV1NmjTR0qVL5eXlZZ0zceJEubu7q3v37srKylLr1q01a9Ysubm5WX3mz5+vJ554wvo0WpcuXQr97iMAAGA/DmOMKe4iSoPMzEw5nU5lZGRc8eOzhk/Nuc5V4UptfK3Pdb8G73fJcSPebwClw5X+/C6xa4gAAABuFAIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwPQIRAACwvRIfiKpVqyaHw5FvGzx4sCSpb9+++Y41bdrUZYzs7GwNHTpU/v7+qlixorp06aKDBw8Wx+0AAIASqMQHog0bNiglJcXali1bJkl66KGHrD4dOnRw6bNkyRKXMYYNG6ZPPvlECxYs0KpVq3Tq1Cl16tRJ586du6H3AgAASib34i7gcqpUqeKy/8orr+i2225TixYtrDYPDw8FBQUVeH5GRoamT5+uuXPnqk2bNpKkefPmKTQ0VMuXL1f79u2vX/EAAKBUKPEzRBfKycnRvHnz1K9fPzkcDqt95cqVCggI0O233664uDilpaVZxzZu3Kjc3Fy1a9fOagsJCVFUVJRWr15d6LWys7OVmZnpsgEAgJtTqQpEixcv1okTJ9S3b1+rLSYmRvPnz9eKFSv0xhtvaMOGDbrnnnuUnZ0tSUpNTVW5cuXk4+PjMlZgYKBSU1MLvdb48ePldDqtLTQ09LrcEwAAKH4l/pHZhaZPn66YmBiFhIRYbT169LD+HBUVpUaNGiksLExffPGFunXrVuhYxhiXWaaLjR49WvHx8dZ+ZmYmoQgAgJtUqQlEv/76q5YvX65FixZdsl9wcLDCwsKUnJwsSQoKClJOTo7S09NdZonS0tLUrFmzQsfx8PCQh4dH0RQPAABKtFLzyGzmzJkKCAhQx44dL9nv2LFjOnDggIKDgyVJDRs2VNmyZa1Pp0lSSkqKtm3bdslABAAA7KNUzBCdP39eM2fO1KOPPip39/8r+dSpUxozZoweeOABBQcHa9++ffrHP/4hf39/3X///ZIkp9Op/v37a/jw4fLz85Ovr69GjBihOnXqWJ86AwAA9lYqAtHy5cu1f/9+9evXz6Xdzc1NW7du1Zw5c3TixAkFBwerVatWWrhwoby8vKx+EydOlLu7u7p3766srCy1bt1as2bNkpub242+FQAAUAKVikDUrl07GWPytXt6euqrr7667Pnly5fX5MmTNXny5OtRHgAAKOVKzRoiAACA64VABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbI9ABAAAbK9U/C4zAChpGj41p7hLwP/a+Fqf4i4BNwFmiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO0RiAAAgO2V6EA0ZswYORwOly0oKMg6bozRmDFjFBISIk9PT7Vs2VLbt293GSM7O1tDhw6Vv7+/KlasqC5duujgwYM3+lYAAEAJVqIDkSTdcccdSklJsbatW7dax1599VW9+eabmjJlijZs2KCgoCC1bdtWJ0+etPoMGzZMn3zyiRYsWKBVq1bp1KlT6tSpk86dO1cctwMAAEog9+Iu4HLc3d1dZoXyGGP01ltv6ZlnnlG3bt0kSbNnz1ZgYKDef/99DRw4UBkZGZo+fbrmzp2rNm3aSJLmzZun0NBQLV++XO3bt7+h9wIAAEqmEj9DlJycrJCQEFWvXl09e/bUL7/8Iknau3evUlNT1a5dO6uvh4eHWrRoodWrV0uSNm7cqNzcXJc+ISEhioqKsvoUJjs7W5mZmS4bAAC4OZXoQNSkSRPNmTNHX331laZNm6bU1FQ1a9ZMx44dU2pqqiQpMDDQ5ZzAwEDrWGpqqsqVKycfH59C+xRm/Pjxcjqd1hYaGlqEdwYAAEqSEh2IYmJi9MADD6hOnTpq06aNvvjiC0l/PBrL43A4XM4xxuRru9iV9Bk9erQyMjKs7cCBA1d5FwAAoKQr0YHoYhUrVlSdOnWUnJxsrSu6eKYnLS3NmjUKCgpSTk6O0tPTC+1TGA8PD3l7e7tsAADg5lSqAlF2drZ27Nih4OBgVa9eXUFBQVq2bJl1PCcnR4mJiWrWrJkkqWHDhipbtqxLn5SUFG3bts3qAwAAUKI/ZTZixAh17txZVatWVVpamsaOHavMzEw9+uijcjgcGjZsmMaNG6eaNWuqZs2aGjdunCpUqKBevXpJkpxOp/r376/hw4fLz89Pvr6+GjFihPUIDgAAQCrhgejgwYN6+OGHdfToUVWpUkVNmzbV2rVrFRYWJkkaOXKksrKyNGjQIKWnp6tJkyZaunSpvLy8rDEmTpwod3d3de/eXVlZWWrdurVmzZolNze34rotAABQwpToQLRgwYJLHnc4HBozZozGjBlTaJ/y5ctr8uTJmjx5chFXBwAAbhalag0RAADA9UAgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtkcgAgAAtude3AUAAFDSNXxqTnGXgP+18bU+12VcZogAAIDtEYgAAIDtEYgAAIDtEYgAAIDtEYgAAIDtlehANH78eDVu3FheXl4KCAhQ165dtWvXLpc+ffv2lcPhcNmaNm3q0ic7O1tDhw6Vv7+/KlasqC5duujgwYM38lYAAEAJVqIDUWJiogYPHqy1a9dq2bJlOnv2rNq1a6fTp0+79OvQoYNSUlKsbcmSJS7Hhw0bpk8++UQLFizQqlWrdOrUKXXq1Ennzp27kbcDAABKqBL9PUQJCQku+zNnzlRAQIA2btyo5s2bW+0eHh4KCgoqcIyMjAxNnz5dc+fOVZs2bSRJ8+bNU2hoqJYvX6727dtfvxsAAAClQomeIbpYRkaGJMnX19elfeXKlQoICNDtt9+uuLg4paWlWcc2btyo3NxctWvXzmoLCQlRVFSUVq9eXei1srOzlZmZ6bIBAICbU6kJRMYYxcfH669//auioqKs9piYGM2fP18rVqzQG2+8oQ0bNuiee+5Rdna2JCk1NVXlypWTj4+Py3iBgYFKTU0t9Hrjx4+X0+m0ttDQ0OtzYwAAoNiV6EdmFxoyZIh+/PFHrVq1yqW9R48e1p+joqLUqFEjhYWF6YsvvlC3bt0KHc8YI4fDUejx0aNHKz4+3trPzMwkFAEAcJMqFTNEQ4cO1WeffaZvvvlGt9566yX7BgcHKywsTMnJyZKkoKAg5eTkKD093aVfWlqaAgMDCx3Hw8ND3t7eLhsAALg5lehAZIzRkCFDtGjRIq1YsULVq1e/7DnHjh3TgQMHFBwcLElq2LChypYtq2XLlll9UlJStG3bNjVr1uy61Q4AAEqPEv3IbPDgwXr//ff16aefysvLy1rz43Q65enpqVOnTmnMmDF64IEHFBwcrH379ukf//iH/P39df/991t9+/fvr+HDh8vPz0++vr4aMWKE6tSpY33qDAAA2FuJDkRTp06VJLVs2dKlfebMmerbt6/c3Ny0detWzZkzRydOnFBwcLBatWqlhQsXysvLy+o/ceJEubu7q3v37srKylLr1q01a9Ysubm53cjbAQAAJVSJDkTGmEse9/T01FdffXXZccqXL6/Jkydr8uTJRVUaAAC4iZToNUQAAAA3AoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYHoEIAADYnq0C0dtvv63q1aurfPnyatiwob777rviLgkAAJQAtglECxcu1LBhw/TMM89o8+bNuvvuuxUTE6P9+/cXd2kAAKCY2SYQvfnmm+rfv78ef/xxRUZG6q233lJoaKimTp1a3KUBAIBiZotAlJOTo40bN6pdu3Yu7e3atdPq1auLqSoAAFBSuBd3ATfC0aNHde7cOQUGBrq0BwYGKjU1tcBzsrOzlZ2dbe1nZGRIkjIzM6/4uueys66iWlwPf+Z9u1q83yUH77e98H7by599v/P6G2Mu2c8WgSiPw+Fw2TfG5GvLM378eL344ov52kNDQ69Lbbi+nJP/Vtwl4Abi/bYX3m97udr3++TJk3I6nYUet0Ug8vf3l5ubW77ZoLS0tHyzRnlGjx6t+Ph4a//8+fM6fvy4/Pz8Cg1RN6PMzEyFhobqwIED8vb2Lu5ycJ3xftsL77e92PX9Nsbo5MmTCgkJuWQ/WwSicuXKqWHDhlq2bJnuv/9+q33ZsmW67777CjzHw8NDHh4eLm2VK1e+nmWWaN7e3rb6B2R3vN/2wvttL3Z8vy81M5THFoFIkuLj4xUbG6tGjRopOjpa7777rvbv36+//Y2pVgAA7M42gahHjx46duyYXnrpJaWkpCgqKkpLlixRWFhYcZcGAACKmW0CkSQNGjRIgwYNKu4yShUPDw+98MIL+R4f4ubE+20vvN/2wvt9aQ5zuc+hAQAA3ORs8cWMAAAAl0IgAgAAtkcgAgAAtkcgAuCiZcuWGjZs2CX7VKtWTW+99dYNqQdFY9asWX/6u9T69u2rrl27Xpd6UHI4HA4tXry4uMsodgQim+E/cPbUt29fORyOAr93a9CgQXI4HOrbt68kadGiRfrnP/95gyvEtSjs3/XKlSvlcDh04sQJ9ejRQz///PONLw4Fyvs36XA4VLZsWYWHh2vEiBE6ffr0Da8lJSVFMTExN/y6JQ2BCLCJ0NBQLViwQFlZ//dLKn///Xd98MEHqlq1qtXm6+srLy+v4igR15Gnp6cCAgKKuwxcoEOHDkpJSdEvv/yisWPH6u2339aIESPy9cvNzb2udQQFBfFRfBGIcIHExETddddd8vDwUHBwsJ5++mmdPXtWkvTf//5XlStX1vnz5yVJSUlJcjgceuqpp6zzBw4cqIcffrhYasflNWjQQFWrVtWiRYustkWLFik0NFT169e32i5+ZJaWlqbOnTvL09NT1atX1/z5829k2SgiBT0yGzt2rAICAuTl5aXHH39cTz/9tO688858577++usKDg6Wn5+fBg8efN1/QNuFh4eHgoKCFBoaql69eql3795avHixxowZozvvvFMzZsxQeHi4PDw8ZIxRRkaGBgwYoICAAHl7e+uee+7Rli1brPEuPK9q1aqqVKmS/v73v+vcuXN69dVXFRQUpICAAL388ssudVz4yOzCWcU8ef+937dvn6T/+7v0+eefKyIiQhUqVNCDDz6o06dPa/bs2apWrZp8fHw0dOhQnTt37nq/jEWGQARJ0m+//aZ7771XjRs31pYtWzR16lRNnz5dY8eOlSQ1b95cJ0+e1ObNmyX9EZ78/f2VmJhojbFy5Uq1aNGiWOrHlXnsscc0c+ZMa3/GjBnq16/fJc/p27ev9u3bpxUrVuijjz7S22+/rbS0tOtdKq6z+fPn6+WXX9aECRO0ceNGVa1aVVOnTs3X75tvvtGePXv0zTffaPbs2Zo1a5ZmzZp14wu2AU9PTyts7t69Wx9++KE+/vhjJSUlSZI6duyo1NRULVmyRBs3blSDBg3UunVrHT9+3Bpjz549+vLLL5WQkKAPPvhAM2bMUMeOHXXw4EElJiZqwoQJevbZZ7V27dprqvXMmTOaNGmSFixYoISEBK1cuVLdunXTkiVLtGTJEs2dO1fvvvuuPvroo2u6zo1kq2+qRuHefvtthYaGasqUKXI4HKpVq5YOHTqkUaNG6fnnn5fT6dSdd96plStXqmHDhlq5cqWefPJJvfjiizp58qROnz6tn3/+WS1btizuW8ElxMbGavTo0dq3b58cDoe+//57LViwQCtXriyw/88//6wvv/xSa9euVZMmTSRJ06dPV2Rk5A2sGlfi888/V6VKlVzaLvV/55MnT1b//v312GOPSZKef/55LV26VKdOnXLp5+PjoylTpsjNzU21atVSx44d9fXXXysuLq7ob8LG1q9fr/fff1+tW7eWJOXk5Gju3LmqUqWKJGnFihXaunWr0tLSrMdbr7/+uhYvXqyPPvpIAwYMkCSdP39eM2bMkJeXl2rXrq1WrVpp165dWrJkicqUKaOIiAhNmDBBK1euVNOmTa+63tzcXE2dOlW33XabJOnBBx/U3LlzdfjwYVWqVMm69jfffKMePXpcy0tzwzBDBEnSjh07FB0dLYfDYbX95S9/0alTp3Tw4EFJfzxKWblypYwx+u6773TfffcpKipKq1at0jfffKPAwEDVqlWruG4BV8Df318dO3bU7NmzNXPmTHXs2FH+/v6F9t+xY4fc3d3VqFEjq61WrVp/+tNKuP5atWqlpKQkl+29994rtP+uXbt01113ubRdvC9Jd9xxh9zc3Kz94OBgZgiLSF6ILV++vKKjo9W8eXNNnjxZkhQWFmaFIUnauHGjTp06JT8/P1WqVMna9u7dqz179lj9qlWr5rIGMDAwULVr11aZMmVc2q71PaxQoYIVhvLGrFatmksoL4rr3EjMEEGSZIxxCUN5bZKs9pYtW2r69OnasmWLypQpo9q1a6tFixZKTExUeno6j8tKiX79+mnIkCGSpH//+9+X7Hvx3wGUXBUrVlSNGjVc2vL+Z6Ywhf2bv1DZsmXznZO3lhDXplWrVpo6darKli2rkJAQl9e6YsWKLn3Pnz+v4ODgAmdzL/wflILerz/zHuYFpwv/LhS0Zuxar1MSMUMESVLt2rW1evVql38Eq1evlpeXl2655RZJ/7eO6K233lKLFi3kcDjUokULrVy5kvVDpUiHDh2Uk5OjnJwctW/f/pJ9IyMjdfbsWf3www9W265du1wWXKJ0ioiI0Pr1613aLnyfcf3lhdiwsLB8YeJiDRo0UGpqqtzd3VWjRg2X7VKzvH9W3qxUSkqK1Za3hulmRyCyoYyMjHxT6wMGDNCBAwc0dOhQ7dy5U59++qleeOEFxcfHW//HkLeOaN68edZaoebNm2vTpk2sHypF3NzctGPHDu3YscPlUUhBIiIi1KFDB8XFxWndunXauHGjHn/8cXl6et6ganG9DB06VNOnT9fs2bOVnJyssWPH6scff2Q2sIRq06aNoqOj1bVrV3311Vfat2+fVq9erWeffbZIg2yNGjUUGhqqMWPG6Oeff9YXX3yhN954o8jGL8kIRDa0cuVK1a9f32V74YUXtGTJEq1fv1716tXT3/72N/Xv31/PPvusy7mtWrXSuXPnrPDj4+Oj2rVrq0qVKiy0LUW8vb3l7e19RX1nzpyp0NBQtWjRQt26dbM+9ovSrXfv3ho9erRGjBihBg0aaO/everbt6/Kly9f3KWhAA6HQ0uWLFHz5s3Vr18/3X777erZs6f27dunwMDAIrtO2bJl9cEHH2jnzp2qV6+eJkyYYH3a+GbnMAU9NAYA2E7btm0VFBSkuXPnFncpwA3HomoAsKEzZ87of/7nf9S+fXu5ubnpgw8+0PLly7Vs2bLiLg0oFswQAYANZWVlqXPnztq0aZOys7MVERGhZ599Vt26dSvu0oBiQSACAAC2x6JqAABgewQiAABgewQiAABgewQiAABgewQiADfUvn375HA4bPPrAACUDgQiADdUaGioUlJSFBUVVdylXJFq1arprbfeKu4yAFxnBCIAN0xOTo7c3NwUFBQkd3e+F/bPKui3jgMoGgQiAFetZcuWGjJkiIYMGaLKlSvLz89Pzz77rPK+3qxatWoaO3as+vbtK6fTqbi4uAIfmW3fvl0dO3aUt7e3vLy8dPfdd2vPnj3W8ZkzZyoyMlLly5dXrVq19Pbbb19xjQcPHlTPnj3l6+urihUrqlGjRlq3bp0kac+ePbrvvvsUGBioSpUqqXHjxlq+fLnL/f3666968skn5XA4XH7x6erVq9W8eXN5enoqNDRUTzzxhE6fPm0dT0lJUceOHeXp6anq1avr/fffzzfbtH//ft13332qVKmSvL291b17dx0+fNg6PmbMGN15552aMWOGwsPD5eHhodmzZ8vPz0/Z2dku9/nAAw+oT58+V/y6AHBFIAJwTWbPni13d3etW7dOkyZN0sSJE/Xee+9Zx1977TVFRUVp48aNeu655/Kd/9tvv6l58+YqX768VqxYoY0bN6pfv346e/asJGnatGl65pln9PLLL2vHjh0aN26cnnvuOc2ePfuytZ06dUotWrTQoUOH9Nlnn2nLli0aOXKkzp8/bx2/9957tXz5cm3evFnt27dX586dtX//fknSokWLdOutt+qll15SSkqKUlJSJElbt25V+/bt1a1bN/34449auHChVq1apSFDhljX7tOnjw4dOqSVK1fq448/1rvvvqu0tDTruDFGXbt21fHjx5WYmKhly5Zpz5496tGjh8s97N69Wx9++KE+/vhjJSUlqXv37jp37pw+++wzq8/Ro0f1+eef67HHHrvsawKgEAYArlKLFi1MZGSkOX/+vNU2atQoExkZaYwxJiwszHTt2tXlnL179xpJZvPmzcYYY0aPHm2qV69ucnJyCrxGaGioef/9913a/vnPf5ro6OjL1vfOO+8YLy8vc+zYsSu+p9q1a5vJkydb+2FhYWbixIkufWJjY82AAQNc2r777jtTpkwZk5WVZXbs2GEkmQ0bNljHk5OTjSRrrKVLlxo3Nzezf/9+q8/27duNJLN+/XpjjDEvvPCCKVu2rElLS3O51t///ncTExNj7b/11lsmPDzc5X0A8OcwQwTgmjRt2tTlUVJ0dLSSk5N17tw5SVKjRo0ueX5SUpLuvvtulS1bNt+xI0eO6MCBA+rfv78qVapkbWPHjnV5pHapsevXry9fX98Cj58+fVojR45U7dq1VblyZVWqVEk7d+60ZogKs3HjRs2aNculpvbt2+v8+fPau3evdu3aJXd3dzVo0MA6p0aNGvLx8bH2d+zYodDQUIWGhlpteXXs2LHDagsLC1OVKlVcrh8XF6elS5fqt99+k/THI8W+ffu6vA8A/hxWNQK4ripWrHjJ456enoUey3u0NW3aNDVp0sTlmJub22WvfamxJempp57SV199pddff101atSQp6enHnzwQeXk5FzyvPPnz2vgwIF64okn8h2rWrWqdu3aVeB55oJfHWmMKTDAXNxe0OtXv3591atXT3PmzFH79u21detW/fe//71kzQAujUAE4JqsXbs2337NmjWvKLBIUt26dTV79mzl5ubmmyUKDAzULbfcol9++UW9e/f+07XVrVtX7733no4fP17gLNF3332nvn376v7775f0x5qiffv2ufQpV66cNduVp0GDBtq+fbtq1KhR4HVr1aqls2fPavPmzWrYsKGkP9YCnThxwupTu3Zt7d+/XwcOHLBmiX766SdlZGQoMjLysvf2+OOPa+LEifrtt9/Upk0bl5kmAH8ej8wAXJMDBw4oPj5eu3bt0gcffKDJkyfr//2//3fF5w8ZMkSZmZnq2bOnfvjhByUnJ2vu3LnWLMuYMWM0fvx4/etf/9LPP/+srVu3aubMmXrzzTcvO/bDDz+soKAgde3aVd9//71++eUXffzxx1qzZo2kPx5jLVq0SElJSdqyZYt69eplzUrlqVatmr799lv99ttvOnr0qCRp1KhRWrNmjQYPHqykpCQlJyfrs88+09ChQyX9EYjatGmjAQMGaP369dq8ebMGDBggT09Pa/anTZs2qlu3rnr37q1NmzZp/fr16tOnj1q0aHHZx4yS1Lt3b/3222+aNm2a+vXrd8WvN4CCEYgAXJM+ffooKytLd911lwYPHqyhQ4dqwIABV3y+n5+fVqxYYX0irGHDhpo2bZo1W/T444/rvffe06xZs1SnTh21aNFCs2bNUvXq1S87drly5bR06VIFBATo3nvvVZ06dfTKK69Ys1cTJ06Uj4+PmjVrps6dO6t9+/Yu634k6aWXXtK+fft02223WWt56tatq8TERCUnJ+vuu+9W/fr19dxzzyk4ONg6b86cOQoMDFTz5s11//33Ky4uTl5eXipfvrwkyeFwaPHixfLx8VHz5s3Vpk0bhYeHa+HChVf0unl7e+uBBx5QpUqV1LVr1ys6B0DhHObCh9oA8Ce0bNlSd955J9/kfAUOHjyo0NBQLV++XK1bty6SMdu2bavIyEhNmjSpSMYD7Iw1RABwHeTNetWpU0cpKSkaOXKkqlWrpubNm1/z2MePH9fSpUu1YsUKTZkypQiqBcAjMwCl1rhx41w++n7hFhMTU6y15ebm6h//+IfuuOMO3X///apSpYpWrlxZ4NcL/FkNGjTQwIEDNWHCBEVERBRBtQB4ZAag1Dp+/LiOHz9e4DFPT0/dcsstN7giAKUVgQgAANgej8wAAIDtEYgAAIDtEYgAAIDtEYgAAIDtEYgAAIDtEYgAAIDtEYgAAIDtEYgAAIDt/X+0VSTMAi2xIQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=y)\n",
    "plt.title(\"Distribution of Price Category\",font = \"cursive\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "01918959-ee11-407b-b384-200994a6c5fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#KM Driven vs Price Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "5af6c2d4-57eb-47b8-b6a1-b1c10c2d9947",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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SqUSXbt2lTLJycmyKRgSExOh1WrRvHnzCt+jUqmEp6en7FGXhISEYNOmTfjggw/w1ltv4YMPPsDGjRtZVBERkWOq+XH0d/fiiy+KJk2aiJ07d4rMzEyxdetW0bhxY/H6669LmcWLFwu1Wi22bt0q0tPTxciRI4Wfn58wGAxSZvz48aJp06Zi9+7d4vDhw+LJJ58UHTt2FKWlpVJm4MCBIigoSOh0OqHT6USHDh1EeHi4tL60tFQEBgaKPn36iMOHD4vdu3eLpk2biqioKClTUFAgfH19xciRI0V6errYunWr8PT0FEuWLKnye64rVwUSERE5kqp+f9u0sDIYDGLKlCmiWbNmws3NTbRo0UK8+eabwmg0Shmz2Syio6OFRqMRSqVShISEiPT0dNl2bt26JaKiooS3t7dQqVQiPDxcZGVlyTJXr14VERERwsPDQ3h4eIiIiAiRn58vy5w/f16EhYUJlUolvL29RVRUlGxqBSGESEtLE48//rhQKpVCo9GIuXPnVnmqBSFYWBEREdmjqn5/K4TgtOH3k8FggFqthl6vr3OnBYmIiOxVVb+/ea9AIiIiIithYUVERERkJSysiIiIiKyEhRURERGRlbCwIiIiIrISFlZEREREVsLCioiIiMhKWFgRERERWQkLKyIiIiIrYWFFREREZCUsrIiIiIishIUVERERkZWwsCIiIiKyEhZWRERERFbCwoqIiIjISlhYEREREVkJCysiIiIiK2FhRURERGQlLKyIiIiIrISFFREREZGVsLAiIiIishIWVkRERERWwsKKiIiIyEpYWBERERFZCQsrIiIiIithYUVERERkJSysiIiIiKyEhRURERGRlbCwIiIiIrISFlZEREREVsLCioiIiMhKbFpYNW/eHAqFotxj4sSJAAAhBObOnQutVguVSoXevXsjIyNDtg2j0YhJkyahcePGcHd3x5AhQ3Dx4kVZJj8/H5GRkVCr1VCr1YiMjERBQYEsk5WVhcGDB8Pd3R2NGzfG5MmTUVxcLMukp6cjNDQUKpUKTZo0wfz58yGEsP4HQ0RERHbJpoXVoUOHkJOTIz127doFABg2bBgA4N1338XSpUuxfPlyHDp0CBqNBv369cP169elbUydOhXbtm1DXFwc9u7dixs3biA8PBwmk0nKjBo1CqmpqUhISEBCQgJSU1MRGRkprTeZTAgL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pk6dijfffBNHjhzB448/jkGDBiErK8vWTSMiIiIbY2F1j5YuXYoxY8bglVdeQdu2bfHhhx/C398fK1eutHXTiIiIyMY4eP0eFBcXIyUlBf/6179ky/v37499+/bZqFVE96aoqKjGe1jPnz8v+29NadasGdzc3Gp0H0T2hL/ftsfC6h788ccfMJlM5a7o8PX1RW5uboWvMRqNMBqN0nODwVCjbawO/iLWLVlZWRg3btx92dfChQtrdPurV6/mVUl/gr/fdQt/v22PhVU1KBQK2XMhRLllFjExMZg3b979aFa18RexbmnWrBlWr15t62ZYRbNmzWzdhFqPv991C3+/bU8hhBC2boS9KC4uRv369fHVV1/hmWeekZZPmTIFqampSEoqP5tuRT1W/v7+0Ov18PT0vC/t/jP34y/a+4V/0RLJ8febyDoMBgPUavWffn+zx+oeuLq6omvXrti1a5essNq1axeGDh1a4WuUSiWUSuX9amK1uLm58a9AIgfF32+i+4uF1T2aNm0aIiMj0a1bNwQHB2P16tXIysrC+PHjbd00IiIisjEWVvdo+PDhuHr1KubPn4+cnBwEBgYiPj7eru+XRERERNbBMVb3WVXP0RIREVHtUdXvb04QSkRERGQlLKyIiIiIrISFFREREZGVsLAiIiIishIWVkRERERWwsKKiIiIyEpYWBERERFZCQsrIiIiIithYUVERERkJSysiIiIiKyE9wq8zyx3EDIYDDZuCREREVWV5Xv7z+4EyMLqPrt+/ToAwN/f38YtISIiont1/fp1qNXqu67nTZjvM7PZjEuXLsHDwwMKhcLWzblvDAYD/P39ceHCBd58ug7g8a5beLzrlrp6vIUQuH79OrRaLZyc7j6Sij1W95mTkxOaNm1q62bYjKenZ536RazreLzrFh7vuqUuHu/KeqosOHidiIiIyEpYWBERERFZCQsrui+USiWio6OhVCpt3RS6D3i86xYe77qFx7tyHLxOREREZCXssSIiIiKyEhZWRERERFbCwoqIakTv3r0xderUSjPNmzfHhx9+eF/aQ9axfv16NGzY8J5eM3r0aDz99NM10h6qPRQKBb755htbN8PmWFhRtfAfyrpp9OjRUCgUGD9+fLl1EyZMgEKhwOjRowEAW7duxdtvv32fW0h/xd1+r/fs2QOFQoGCggIMHz4cp06duv+N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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=y,y = data[\"km_driven\"])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "dfc2722e-5d52-4161-a2bc-61c21317904b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Owner vs Price Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "fb741e83-19d6-4e0e-8404-fe3e9df75189",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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JSQkWFWrVrUyMzPtffbs2WNJsnbu3GlZlmWNHz/e8vb2to4ePerwWk899ZTVtWtX+/OZM2dajRo1cjgPAMqOmSUAFerOO+90uDwVExOjffv2qbCwUJLUunVr4/iUlBTFxsbK29u7xLZffvlFWVlZGjhwoGrUqGF/TJo0yeEynWnfLVq0UO3atUvdfvr0ab3wwguKjo5WzZo1VaNGDe3du9c+s3Qpu3fv1pIlSxxq6ty5s4qKipSRkaH09HR5eXmpZcuW9jGRkZGqVauW/XlaWprCwsIUFhZmbyuuIy0tzd4WHh6uunXrOrz+4MGDlZCQoMOHD0u6eJmyf//+DucBQNmxghKAW91www3G7X5+fpfcVny5bP78+WrTpo3DtqpVq172tU37lqTnn39e//73vzV9+nRFRkbKz89PDz/8sM6dO2ccV1RUpCeffFIjRowosa1BgwZKT08vdZz1u981tyyr1HDzx/bS3r8WLVro1ltv1fvvv6/OnTvru+++0yeffGKsGcClEZYAVKjt27eXeH7zzTeXKcxI0i233KKlS5fq/PnzJWaXQkJCdOONN+qnn35Snz59yl3bLbfcogULFuj48eOlzi5t3bpV/fv3V8+ePSVdXMN04MABhz7VqlWzz5IVa9mypfbs2aPIyMhSX7dp06a6cOGCkpOT1apVK0kX1x6dPHnS3ic6OlqZmZnKysqyzy59//33ys3NVVRU1GWPbdCgQZoxY4YOHz6sjh07OsxQASgfLsMBqFBZWVkaPXq00tPTtWLFCr399tt69tlnyzx+2LBhysvLU+/evbVr1y7t27dPy5Yts8/OTJgwQVOmTNE//vEP/fDDD/ruu++0ePFivfXWW5fd92OPPabQ0FD16NFDX331lX766SetWrVKSUlJki5eGlu9erVSUlKUmpqqxx9/3D6bVaxhw4b64osvdPjwYR07dkySNGbMGCUlJemZZ55RSkqK9u3bp3Xr1mn48OGSLoaljh07asiQIdq5c6eSk5M1ZMgQ+fn52WeNOnbsqFtuuUV9+vTRN998o507d6pfv36Ki4u77KVLSerTp48OHz6s+fPna8CAAWV+vwGURFgCUKH69euns2fP6o477tAzzzyj4cOHa8iQIWUeX6dOHW3atMn+zbVWrVpp/vz59lmmQYMGacGCBVqyZImaN2+uuLg4LVmyRBEREZfdd7Vq1ZSQkKDg4GB169ZNzZs315tvvmmf9ZoxY4Zq1aqltm3b6r777lPnzp0d1hlJ0sSJE3XgwAE1btzYvnbolltu0ZYtW7Rv3z7FxsaqRYsWeuWVV1SvXj37uPfff18hISFq166devbsqcGDB8vf31++vr6SJJvNprVr16pWrVpq166dOnbsqEaNGmnlypVlet8CAgL00EMPqUaNGurRo0eZxgAonc36/UVyAHCh9u3b67bbbuMO12Vw6NAhhYWFaePGjbr77rtdss977rlHUVFRmjVrlkv2B1yvWLMEAG5QPFvWvHlzZWdn64UXXlDDhg3Vrl27K9738ePHlZCQoE2bNmn27NkuqBa4vnEZDsA1a/LkyQ5f3//9o2vXrm6t7fz583rppZf05z//WT179lTdunWVmJhY6i0Syqtly5Z68sknNXXqVDVp0sQF1QLXNy7DAbhmHT9+XMePHy91m5+fn2688carXBEAT0RYAgAAMOAyHAAAgAFhCQAAwICwBAAAYEBYAgAAMCAsAQAAGBCWAAAADAhLAAAABoQlAAAAg/8DUUhJjBYpxicAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=y, y=data['owner_count'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12493713-0b31-4e10-bfbf-c6a11ce38ba0",
   "metadata": {},
   "source": [
    "# SVM MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "88687159-2f1c-4723-81c7-71eb69c50ab0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q.4 Split the dataset into training and testing sets. Train a SVM model and display accuracy Score, Confusion Matrix, Classification report.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7499a230-5b72-43fa-bdac-213a93b6201b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score,confusion_matrix,classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "dad735b5-11df-4899-859f-8a3cd0465097",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "e922b40e-524d-477d-a199-0e7485026ba1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = SVC()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "ad30fc85-64d6-47cd-9e49-2ac02250892e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-4 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-4 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-4 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-4 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-4 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-4 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-4 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-4 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-4 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-4 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-4 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-4 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-4 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;SVC<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html\">?<span>Documentation for SVC</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>SVC()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "SVC()"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "a94ad89b-e3bf-45e2-bcd2-dcaacc6925eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Low', 'Mid', 'High', 'High', 'High', 'Low', 'High', 'Mid', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Mid', 'High', 'Mid', 'Low', 'Mid', 'Low',\n",
       "       'Mid', 'Low', 'Low', 'Low', 'Mid', 'Low', 'Mid', 'Low', 'Low',\n",
       "       'Mid', 'Low', 'High', 'High', 'Mid', 'Mid', 'Mid', 'Mid', 'Low',\n",
       "       'Mid', 'Low', 'Low', 'Low', 'Low', 'Low', 'High', 'Low', 'Mid',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Mid', 'Low', 'Low', 'Premium', 'Low',\n",
       "       'High', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'High', 'Mid', 'Low',\n",
       "       'High', 'High', 'Low', 'Mid', 'Mid', 'Mid', 'Mid', 'Mid', 'Mid',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Mid', 'High', 'Low', 'Mid', 'Low',\n",
       "       'Mid', 'Mid', 'Premium', 'Low', 'High', 'Low', 'Premium', 'Low',\n",
       "       'Mid', 'High', 'Low', 'Low', 'Low', 'Low', 'Mid', 'Premium', 'Low',\n",
       "       'Mid', 'Low', 'Low', 'Premium', 'Low', 'Low', 'Low', 'Low', 'Mid',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Mid', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Mid', 'Mid', 'Low', 'Mid', 'Low', 'Low', 'High', 'Low',\n",
       "       'Mid', 'Low', 'Low', 'Mid', 'Low', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Mid', 'Mid', 'Low', 'Low', 'High', 'Mid',\n",
       "       'Low', 'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Premium', 'Mid',\n",
       "       'Premium', 'Mid', 'Low', 'High', 'Low', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Low', 'High', 'Premium', 'Mid',\n",
       "       'High', 'Mid', 'Mid', 'Mid', 'Low', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Low', 'High', 'High', 'Mid', 'Low', 'Low', 'High', 'High', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Low', 'Low', 'Mid',\n",
       "       'Mid', 'Low', 'Mid', 'Low', 'Low', 'Mid', 'Mid', 'High', 'Low',\n",
       "       'High', 'Low', 'Low', 'Premium', 'Low', 'Low', 'Low', 'Premium',\n",
       "       'Low', 'Mid', 'Premium', 'High', 'Low', 'Low', 'Low', 'Low',\n",
       "       'High', 'Mid', 'Premium', 'Low', 'Low', 'Low', 'Mid', 'High',\n",
       "       'Low', 'Premium', 'High', 'High', 'Low', 'Low', 'High', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Mid', 'Low', 'High', 'Mid', 'High',\n",
       "       'Low', 'High', 'Mid', 'Low', 'Low', 'Low', 'Low', 'Low', 'Mid',\n",
       "       'High', 'Low', 'Premium', 'Low', 'Mid', 'Mid', 'Low', 'Low', 'Mid',\n",
       "       'Mid', 'Mid', 'High', 'Mid', 'Low', 'Low', 'High', 'Mid', 'Low',\n",
       "       'Low', 'High', 'Low', 'Premium', 'Low', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Mid', 'Low', 'Mid', 'Premium', 'Low', 'Mid', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'Low', 'Mid', 'Low', 'Premium', 'Low', 'Low', 'Low', 'Mid',\n",
       "       'Low', 'Mid', 'Low', 'High', 'Low', 'High', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Low', 'High', 'Low', 'High', 'Mid', 'Low', 'Low', 'High',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'Mid', 'Mid', 'Mid', 'Premium',\n",
       "       'Low', 'Premium', 'Low', 'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid',\n",
       "       'Premium', 'Low', 'Low', 'Mid', 'Low', 'Low', 'High', 'Low', 'Low',\n",
       "       'Low', 'Low', 'High', 'Low', 'Low', 'Mid', 'Low', 'High', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'Mid', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Premium', 'Low',\n",
       "       'Low', 'Low', 'Mid', 'Mid', 'High', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Mid', 'Mid', 'Low', 'Low', 'Low', 'High', 'Mid', 'High',\n",
       "       'Mid', 'Premium', 'Premium', 'Low', 'Mid', 'Low', 'Mid', 'High',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'High', 'Mid', 'High', 'Low', 'Mid',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Premium',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'High', 'Mid', 'Low', 'High',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Mid', 'Low', 'Mid', 'Mid', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Mid', 'High', 'Low', 'Mid', 'High',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Mid', 'Mid', 'Low',\n",
       "       'Mid', 'Premium', 'Mid', 'Mid', 'Low', 'Premium', 'Mid', 'Mid',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Low', 'Premium', 'Low', 'Low', 'Low',\n",
       "       'Low', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'Low', 'High',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Mid', 'Mid', 'High', 'Low', 'High',\n",
       "       'Low', 'High', 'Low', 'Mid', 'Low', 'Low', 'Mid', 'Low', 'High',\n",
       "       'High', 'Low', 'Low', 'Low', 'High', 'Low', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'High', 'Low', 'Mid', 'Premium', 'High', 'Low', 'Mid',\n",
       "       'Mid', 'Low', 'Low', 'Low', 'Low', 'High', 'Low', 'Premium', 'Low',\n",
       "       'Low', 'Mid', 'High', 'High', 'Mid', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Low', 'High', 'Mid', 'Premium', 'Low', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'High', 'Low', 'High', 'High', 'High', 'Low', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'High', 'Mid', 'Mid', 'Mid', 'Premium',\n",
       "       'High', 'High', 'Mid', 'Low', 'Low', 'Low', 'Premium', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Low', 'Mid', 'Low', 'Mid', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Mid', 'High', 'Low', 'Low', 'Low', 'Low',\n",
       "       'Mid', 'Low', 'Mid', 'High', 'Low', 'High', 'Mid', 'Premium',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Low', 'Low', 'High', 'Low', 'Mid',\n",
       "       'Low', 'Low', 'High', 'Mid', 'Low', 'Low', 'Mid', 'High', 'Low',\n",
       "       'Low', 'Mid', 'High', 'High', 'Mid', 'Mid', 'Mid', 'Mid', 'Mid',\n",
       "       'Mid', 'Mid', 'Low', 'High', 'Low', 'Low', 'Low', 'Mid', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'Premium', 'High', 'Low',\n",
       "       'High', 'Mid', 'Low', 'High', 'Low', 'Premium', 'Low', 'Mid',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Low', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Mid', 'Low', 'Premium', 'Mid', 'Low', 'Low', 'Low', 'High', 'Mid',\n",
       "       'Low', 'Low', 'Mid', 'Low', 'Mid', 'Low', 'Premium', 'Low', 'Low',\n",
       "       'Mid', 'Premium', 'Mid', 'Low', 'High', 'Mid', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'Mid', 'Low', 'High', 'High',\n",
       "       'Low', 'Mid', 'High', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Mid',\n",
       "       'Premium', 'Mid', 'Mid', 'Low', 'Low', 'Low', 'High', 'High',\n",
       "       'Low', 'Mid', 'Low', 'High', 'Low', 'Low', 'Mid', 'Low', 'Low',\n",
       "       'Low', 'High', 'Mid', 'Low', 'Mid', 'Mid', 'Mid', 'Mid', 'Low',\n",
       "       'Mid', 'Low', 'Mid', 'Low', 'Low', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Mid', 'High', 'Low', 'Mid', 'Low', 'Low', 'Low', 'Mid', 'Low',\n",
       "       'High', 'Mid', 'Low', 'Low', 'High', 'Mid', 'Low', 'Premium',\n",
       "       'Mid', 'Low', 'Premium', 'Low', 'Mid', 'Low', 'Premium', 'Premium',\n",
       "       'Premium', 'Low', 'Mid', 'High', 'Low', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Low', 'Mid', 'Mid', 'Premium', 'Mid', 'High', 'Mid', 'Low',\n",
       "       'Low', 'Low', 'Low', 'Low', 'Low', 'Mid', 'High', 'Mid', 'Mid',\n",
       "       'Mid', 'Mid', 'Mid', 'Premium', 'Low', 'Low', 'Premium', 'Low',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'High', 'Premium', 'Low', 'Low', 'Low',\n",
       "       'Mid', 'Low', 'High', 'Low', 'Low', 'Low', 'Low', 'High', 'Mid',\n",
       "       'Mid', 'Low', 'Low', 'Mid', 'Low', 'Premium', 'Low', 'Premium',\n",
       "       'Low', 'Low', 'Low', 'Mid', 'Low', 'Mid', 'Low', 'Low', 'Low',\n",
       "       'Low', 'Low', 'Low', 'High', 'Low', 'Mid', 'Low', 'Low', 'Mid',\n",
       "       'Low', 'Mid', 'Low', 'Low', 'Mid', 'Low', 'High', 'Low', 'Mid',\n",
       "       'Premium', 'Mid', 'Low', 'Mid', 'Mid', 'Low', 'Low', 'High',\n",
       "       'High', 'Low', 'Mid', 'High', 'Premium', 'Mid', 'Low', 'High',\n",
       "       'Mid', 'Low', 'Low', 'Mid', 'Low', 'Low'], dtype=object)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ypri = model.predict(x_test)\n",
    "ypri"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "e5bba58e-8277-4e28-89d9-bbb3f334bb46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy 85.36866359447005\n",
      "Confusion metrics [[102  10   9   0]\n",
      " [  9 384  22   2]\n",
      " [  0  61 205   0]\n",
      " [  4  10   0  50]]\n"
     ]
    }
   ],
   "source": [
    "print(\"SVM Accuracy\",accuracy_score(y_test,ypri)*100)\n",
    "print(\"Confusion metrics\",confusion_matrix(y_test,ypri))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "08733e6b-913a-47ed-b63d-df6615a6faee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "j = confusion_matrix(y_test,ypri)\n",
    "sns.heatmap(j,annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "a3afa888-4de6-4f4e-a6f1-f0d75b3d1f6e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "classification_report               precision    recall  f1-score   support\n",
      "\n",
      "        High       0.89      0.84      0.86       121\n",
      "         Low       0.83      0.92      0.87       417\n",
      "         Mid       0.87      0.77      0.82       266\n",
      "     Premium       0.96      0.78      0.86        64\n",
      "\n",
      "    accuracy                           0.85       868\n",
      "   macro avg       0.89      0.83      0.85       868\n",
      "weighted avg       0.86      0.85      0.85       868\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"classification_report\",classification_report(y_test,ypri))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78c37406-ac28-4ca2-b787-11fad619a083",
   "metadata": {},
   "source": [
    "# RANDOM FOREST MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "eccb8a18-201e-47fe-a6b5-728212239af9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Q.5. Train a Random Forest Classifier on the same data and predict price_category, find - accuracy Score, Confusion Matrix, Classification report. Compare both models accuracy and print the best model for Prediction. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "33c2920f-12b3-4024-b5de-145fc8fa77ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "a7899f21-8f20-4299-ae07-c7a144f320fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "model2 = RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "d8a77610-e608-4193-897b-ed668cf1f40e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-5 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-5 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-5 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-5 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-5 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-5 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-5 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-5 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-5 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-5 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-5 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-5 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-5 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;RandomForestClassifier<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html\">?<span>Documentation for RandomForestClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>RandomForestClassifier()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "77cb7284-f57d-43ad-9539-b5d1c5f7aff4",
   "metadata": {},
   "outputs": [],
   "source": [
    "ypri2 = model2.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "477d5db8-c7fe-4e51-8f37-00bc9242af68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy 91.58986175115207\n",
      "Confusion metrics [[114   2   3   2]\n",
      " [ 16 369  29   3]\n",
      " [  6   7 253   0]\n",
      " [  4   1   0  59]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Random Forest Accuracy\",accuracy_score(y_test,ypri2)*100)\n",
    "print(\"Confusion metrics\",confusion_matrix(y_test,ypri2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "398218fb-56aa-4bdb-a908-275af1fcc2f6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "d = confusion_matrix(y_test,ypri2)\n",
    "sns.heatmap(d,annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "b7e16786-0b13-444c-9a89-f08ae234c64c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "classification_report               precision    recall  f1-score   support\n",
      "\n",
      "        High       0.81      0.94      0.87       121\n",
      "         Low       0.97      0.88      0.93       417\n",
      "         Mid       0.89      0.95      0.92       266\n",
      "     Premium       0.92      0.92      0.92        64\n",
      "\n",
      "    accuracy                           0.92       868\n",
      "   macro avg       0.90      0.93      0.91       868\n",
      "weighted avg       0.92      0.92      0.92       868\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"classification_report\",classification_report(y_test,ypri2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "89bdd497-960c-4ad6-8b8d-f6ae2c6c1d1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compare Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "1c429b8d-5348-4066-a768-7ed60701ef27",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random forest is good model\n"
     ]
    }
   ],
   "source": [
    "SVM = accuracy_score(y_test,ypri)\n",
    "RF = accuracy_score(y_test,ypri2)\n",
    "\n",
    "if RF > SVM:\n",
    "    print(\"Random forest is good model\")\n",
    "else:\n",
    "    print(\"SVM is good model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53c5e6a9-25ef-4b1b-8410-f5c550390348",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
