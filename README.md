# Federated-Machine-Learning-Model-
1. Install 64 bit Python V3.8
2. Install tensorflow and other required package
3. Make sure the data path in line 25 in TFClass.py and TFClass2.py, TFClass_1.py and TFClass2_1.py
    is r'BTC Dataset (with Sentiment).csv'
4. Run Server.py-->it will output the weights update for client1 and client2, client1_1 and client2_1 for each    iteration
5. Run Client1.py, Client2.py,Client1_1.py, Client2_1.py-> it will output the accuracy for each client.
6. Currently the stop criteria has not be been set up. It will continue to run. Final accuracy will    be approaching 1 eventually and weights converge (weights do not change at local client and the    weights of two client are close to each other).

