import json
from scipy import signal
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from  sklearn.ensemble.forest  import RandomForestClassifier
from sklearn.model_selection import train_test_split#将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier#利用邻近点方式训练数据
from sklearn.externals import joblib



# 由于文件中有多行，直接读取会出现错误，因此一行一行读取
if __name__ == '__main__':
    file1 = open("activity1.json", 'r', encoding='utf-8')
    file2 = open("activity2.json", 'r', encoding='utf-8')
    file3 = open("activity3.json", 'r', encoding='utf-8')
    file4 = open("activity4.json", 'r', encoding='utf-8')
    dataset=[]
    dataset1 = []
    dataset2 = []
    dataset3 = []
    dataset4 = []
    for line in file1.readlines():
        lines= json.loads(line)
        b, a = signal.butter(8, 0.2, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        accxData = signal.filtfilt(b, a, lines["accx"])  # data为要过滤的信号
        accxData=accxData.tolist()
        accyData = signal.filtfilt(b, a, lines["accy"])  # data为要过滤的信号
        accyData=accyData.tolist()
        acczData = signal.filtfilt(b, a, lines["accz"])  # data为要过滤的信号
        acczData=acczData.tolist()
        gryxData = signal.filtfilt(b, a, lines["gryx"])  # data为要过滤的信号
        gryxData = gryxData.tolist()
        gryyData = signal.filtfilt(b, a, lines["gryy"])  # data为要过滤的信号
        gryyData = gryyData.tolist()
        gryzData = signal.filtfilt(b, a, lines["gryz"])  # data为要过滤的信号
        gryzData = gryzData.tolist()
        #print(filtedData)

        for i in range(50, 1010, 64):
            xc = accxData[i:i + 128]
            yc = accyData[i:i + 128]
            zc = acczData[i:i + 128]
            xw = gryxData[i:i + 128]
            yw = gryyData[i:i + 128]
            zw = gryzData[i:i + 128]
            #print(len(xc))
            ##这里是一个数据的36个特征值，最后一个1表示label
            s=[max(xc),min(xc),np.median(xc),np.mean(xc),np.std(xc),stats.median_absolute_deviation(xc),max(yc),min(yc),np.median(yc),np.mean(yc),np.std(yc),stats.median_absolute_deviation(yc),max(zc),min(zc),np.median(zc),np.mean(zc),np.std(zc),stats.median_absolute_deviation(zc),max(xw),min(xw),np.median(xw),np.mean(xw),np.std(xw),stats.median_absolute_deviation(xw),max(yw),min(yw),np.median(yw),np.mean(yw),np.std(yw),stats.median_absolute_deviation(yw),max(zw),min(zw),np.median(zw),np.mean(zw),np.std(zw),stats.median_absolute_deviation(zw),1]
            dataset1.append(s)
    for line in file2.readlines():
        lines= json.loads(line)
        b, a = signal.butter(8, 0.2, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        accxData = signal.filtfilt(b, a, lines["accx"])  # data为要过滤的信号
        accxData=accxData.tolist()
        accyData = signal.filtfilt(b, a, lines["accy"])  # data为要过滤的信号
        accyData=accyData.tolist()
        acczData = signal.filtfilt(b, a, lines["accz"])  # data为要过滤的信号
        acczData=acczData.tolist()
        gryxData = signal.filtfilt(b, a, lines["gryx"])  # data为要过滤的信号
        gryxData = gryxData.tolist()
        gryyData = signal.filtfilt(b, a, lines["gryy"])  # data为要过滤的信号
        gryyData = gryyData.tolist()
        gryzData = signal.filtfilt(b, a, lines["gryz"])  # data为要过滤的信号
        gryzData = gryzData.tolist()
        for i in range(50, 1010, 64):
            xc = accxData[i:i + 128]
            yc = accyData[i:i + 128]
            zc = acczData[i:i + 128]
            xw = gryxData[i:i + 128]
            yw = gryyData[i:i + 128]
            zw = gryzData[i:i + 128]
            #print(len(xc))
            ##这里是一个数据的36个特征值，最后一个1表示label
            s=[max(xc),min(xc),np.median(xc),np.mean(xc),np.std(xc),stats.median_absolute_deviation(xc),max(yc),min(yc),np.median(yc),np.mean(yc),np.std(yc),stats.median_absolute_deviation(yc),max(zc),min(zc),np.median(zc),np.mean(zc),np.std(zc),stats.median_absolute_deviation(zc),max(xw),min(xw),np.median(xw),np.mean(xw),np.std(xw),stats.median_absolute_deviation(xw),max(yw),min(yw),np.median(yw),np.mean(yw),np.std(yw),stats.median_absolute_deviation(yw),max(zw),min(zw),np.median(zw),np.mean(zw),np.std(zw),stats.median_absolute_deviation(zw),2]
            dataset2.append(s)
    for line in file3.readlines():
        lines= json.loads(line)
        b, a = signal.butter(8, 0.2, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        accxData = signal.filtfilt(b, a, lines["accx"])  # data为要过滤的信号
        accxData=accxData.tolist()
        accyData = signal.filtfilt(b, a, lines["accy"])  # data为要过滤的信号
        accyData=accyData.tolist()
        acczData = signal.filtfilt(b, a, lines["accz"])  # data为要过滤的信号
        acczData=acczData.tolist()
        gryxData = signal.filtfilt(b, a, lines["gryx"])  # data为要过滤的信号
        gryxData = gryxData.tolist()
        gryyData = signal.filtfilt(b, a, lines["gryy"])  # data为要过滤的信号
        gryyData = gryyData.tolist()
        gryzData = signal.filtfilt(b, a, lines["gryz"])  # data为要过滤的信号
        gryzData = gryzData.tolist()
        #print(filtedData)
       # y = acczData
       # x = range(len(y))
        #plt.plot(y)
       # plt.show()

        for i in range(50, 1010, 64):
            temp=[]
            xc = accxData[i:i + 128]
            yc = accyData[i:i + 128]
            zc = acczData[i:i + 128]
            xw = gryxData[i:i + 128]
            yw = gryyData[i:i + 128]
            zw = gryzData[i:i + 128]
            #print(len(xc))
            ##这里是一个数据的36个特征值，最后一个1表示label
            s=[max(xc),min(xc),np.median(xc),np.mean(xc),np.std(xc),stats.median_absolute_deviation(xc),max(yc),min(yc),np.median(yc),np.mean(yc),np.std(yc),stats.median_absolute_deviation(yc),max(zc),min(zc),np.median(zc),np.mean(zc),np.std(zc),stats.median_absolute_deviation(zc),max(xw),min(xw),np.median(xw),np.mean(xw),np.std(xw),stats.median_absolute_deviation(xw),max(yw),min(yw),np.median(yw),np.mean(yw),np.std(yw),stats.median_absolute_deviation(yw),max(zw),min(zw),np.median(zw),np.mean(zw),np.std(zw),stats.median_absolute_deviation(zw),3]
            dataset3.append(s)
    for line in file4.readlines():
        lines= json.loads(line)
        b, a = signal.butter(8, 0.2, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        accxData = signal.filtfilt(b, a, lines["accx"])  # data为要过滤的信号
        accxData=accxData.tolist()
        accyData = signal.filtfilt(b, a, lines["accy"])  # data为要过滤的信号
        accyData=accyData.tolist()
        acczData = signal.filtfilt(b, a, lines["accz"])  # data为要过滤的信号
        acczData=acczData.tolist()
        gryxData = signal.filtfilt(b, a, lines["gryx"])  # data为要过滤的信号
        gryxData = gryxData.tolist()
        gryyData = signal.filtfilt(b, a, lines["gryy"])  # data为要过滤的信号
        gryyData = gryyData.tolist()
        gryzData = signal.filtfilt(b, a, lines["gryz"])  # data为要过滤的信号
        gryzData = gryzData.tolist()
        #print(filtedData)
       # y = acczData
       # x = range(len(y))
        #plt.plot(y)
       # plt.show()

        for i in range(50, 1010, 64):
            temp=[]
            xc = accxData[i:i + 128]
            yc = accyData[i:i + 128]
            zc = acczData[i:i + 128]
            xw = gryxData[i:i + 128]
            yw = gryyData[i:i + 128]
            zw = gryzData[i:i + 128]
            #print(len(xc))
            ##这里是一个数据的36个特征值，最后一个1表示label
            s=[max(xc),min(xc),np.median(xc),np.mean(xc),np.std(xc),stats.median_absolute_deviation(xc),max(yc),min(yc),np.median(yc),np.mean(yc),np.std(yc),stats.median_absolute_deviation(yc),max(zc),min(zc),np.median(zc),np.mean(zc),np.std(zc),stats.median_absolute_deviation(zc),max(xw),min(xw),np.median(xw),np.mean(xw),np.std(xw),stats.median_absolute_deviation(xw),max(yw),min(yw),np.median(yw),np.mean(yw),np.std(yw),stats.median_absolute_deviation(yw),max(zw),min(zw),np.median(zw),np.mean(zw),np.std(zw),stats.median_absolute_deviation(zw),4]
            dataset4.append(s)
    dataset1=  np.array(dataset1)
    dataset2 = np.array(dataset2)
    dataset3 = np.array(dataset3)
    dataset4 = np.array(dataset4)
    #print(dataset1.shape)
    #print(dataset2.shape)
    #print(dataset3.shape)
    #print(dataset4.shape)
    dataset=np.vstack((dataset1,dataset2,dataset3,dataset4))
   # dataset = pd.DataFrame({'RISK': dataset})
   # dataset.to_csv('dataset.csv', index=False)
    #pd_data = pd.DataFrame(dataset)
   # print(pd_data)
    #pd_data.to_csv('dataset.csv',index=0,header=0)
   # print(dataset.shape)
    y_data=dataset[:,-1]
    x_data=np.delete(dataset,-1,axis=1)
    #print(y_data)
    #print(x_data.shape)
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.3)  # 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
    #knn = KNeighborsClassifier()  # 引入训练方法


    #knn.fit(X_train, y_train)  # 进行填充测试数据进行训练
    #print(knn.predict(X_test))  # 预测特征值
   # x_predit=knn.predict(X_test)
   # print(y_test)  # 真实特征值
    ##for i in range(len(y_test)-1):
        #if x_predit[i]==y_test[i]:
         #   num+=1
# k=num/len(y_test)
   # print(k)
    rfc = RandomForestClassifier()  # 实例化
    rfc = rfc.fit(x_data, y_data)  # 用训练集数据训练模型
    #result = rfc.score(X_test, y_test)
    #knn=KNeighborsClassifier()
    #knn=knn.fit(x_data,y_data)rfc
    #result=knn.predict(X_test)
    #print(result)
    #print(rfc.predict(X_test))
    joblib.dump(rfc, "./cgrfc.pkl")  # lr是训练好的模型， "./ML/test.pkl"是模型要保存的路径及保存模型的文件名，其中，'pkl' 是sklearn中默认的保存格式gai


