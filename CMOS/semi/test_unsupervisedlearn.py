from CMOS.semi.dataloader import load_Dataset_ellipticPlusPlus
from unsupervisedlearn import slefTrain_SVC, slefTrain_AdaBoost, \
    slefTrain_gbBoost, slefTrain_RandomForest, slefTrain_xgboost
from util import myplot

preditcList_acc = []
preditcList_f1_N = []
preditcList_f1_p = []
preditcList_f1 = []
preditcList_precision_N = []
preditcList_precision_P = []
preditcList_recall_N = []
preditcList_recall_P = []

# class Test(TestCase):
#
#     def test_my_label_spread(self):
#
#         for i in range(43, 50):
#             # f1_n_list, timeList = [], [i for i in range(16 + i, 31 + i)]
#             # print("时间片", timeList)
#             data = load_Dataset_ellipticPlusPlus(data_timeList=[i])
#             data_df = data.data
#             f1_n = myLabelSpread(data_df, time=[i])
#
#         # myplot(timeList, f1_n_list, label1='f1_n', title='label_spread结果')
#         # f1_n_list, x = [], [i for i in range(16, 50)]
#         # for time in timeList:
#         #     print("时间片", time)
#         #     data = load_Dataset_ellipticPlusPlus(data_timeList=[time], lasttime=True)
#         #     data_df = data.data_last
#         #     f1_n = myLabelSpread(data_df,time=time)
#         #     f1_n_list.append(f1_n)
#         # myplot(timeList, f1_n_list, label1='f1_n', title='label_spread结果')
#
#
#
#
#     def test_get_predict(self):
#         # for i in range(34, 35):
#         #     f1_n_list, timeList = [], [i for i in range(1 + i, 16 + i)]
#         preditcList_f1_N = []
#         for i in range(1, 50):
#             f1_n_list, timeList = [], [i]
#
#             print("时间片", timeList)
#
#             data = load_Dataset_ellipticPlusPlus(data_timeList=timeList)
#             data_df = data.data
#             # f1_n = myLabelSpread(data_df, time=timeList)
#
#             # acc,f1_macro_N,f1_macro_p,f1_macro,precision_N,precision_P,recall_N,recall_P = slefTrain_DecisionTree(data_df, time=timeList)
#             f1_n = slefTrain_SVC(data_df, time=timeList)
#             # f1_n = slefTrain_AdaBoost(data_df, time=timeList)
#             # f1_n = slefTrain_gbBoost(data_df, time=timeList)
#             #             # f1_n = slefTrain_GaussianProcess(data_df, time=timeList)
#             # f1_n = slefTrain_RandomForest(data_df, time=timeList)
#             # f1_n = slefTrain_xgboost(data_df,time=timeList)
#
#
#             preditcList_f1_N.append(f1_n)
#
#
#             print()
#
#         print("f1负", preditcList_f1_N)
#
#         myplot(x=timeList, y1=preditcList_f1_N, title="selftrain")


if __name__ == "__main__":

    from concurrent.futures import ProcessPoolExecutor, as_completed


    def run_model(time_index, func, data_df, timeList):
        result = func(data_df, time=timeList)
        return time_index, func.__name__, result


    def test_get_predict(self):
        models = [
            slefTrain_SVC,
            slefTrain_AdaBoost,
            slefTrain_gbBoost,
            slefTrain_RandomForest,
            slefTrain_xgboost
        ]

        time_range = range(1, 50)
        results_by_time = {}

        all_tasks = []
        with ProcessPoolExecutor(max_workers=8) as executor:  # 可根据CPU核数调节
            for i in time_range:
                timeList = [i]
                data = load_Dataset_ellipticPlusPlus(data_timeList=timeList)
                data_df = data.data

                for model in models:
                    fut = executor.submit(run_model, i, model, data_df, timeList)
                    all_tasks.append(fut)

            for future in as_completed(all_tasks):
                time_index, model_name, f1_n = future.result()
                results_by_time.setdefault(time_index, {})[model_name] = f1_n
                print(f"✅ 时间片 {time_index} - 模型 {model_name} 完成，结果: {f1_n}")

        # 例如画 SVC 的结果
        svc_results = [results_by_time[t]['slefTrain_SVC'] for t in time_range]
        myplot(x=list(time_range), y1=svc_results, title="selftrain SVC")
