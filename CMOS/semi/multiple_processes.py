from concurrent.futures import ProcessPoolExecutor, as_completed


from .unsupervisedlearn import slefTrain_DecisionTree, slefTrain_SVC, slefTrain_AdaBoost, \
    slefTrain_gbBoost, slefTrain_RandomForest, slefTrain_xgboost


def run_model(time_index, func, timeList):

    import pandas as pd
    data_df = pd.read_csv(f'./temp/temp{timeList[0]}.csv', dtype=float)
    result = func(data_df, time=timeList)
    return time_index, func.__name__, result


def test_get_predict(ellipticplusplus):
    # semi_model list
    models = [
        slefTrain_SVC,
        slefTrain_AdaBoost,
        slefTrain_gbBoost,
        slefTrain_RandomForest,
        slefTrain_xgboost,
        slefTrain_DecisionTree
    ]

    time_range = range(1, 50)

    results_by_time = {}

    all_tasks = []
    BATCH_SIZE = 2  # How many time slices of tasks to submit at a time
    with ProcessPoolExecutor(max_workers=6) as executor:
        for batch_start in range(0, len(time_range), BATCH_SIZE):
            batch_times = list(time_range)[batch_start: batch_start + BATCH_SIZE]
            all_tasks = []

            for i in batch_times:
                timeList = [i]
                print(f"loading {timeList} data...")
                ellipticplusplus.get_Sliceoftimeofdatadf(i)
                data_df = ellipticplusplus.data
                data_df.to_csv(path_or_buf=f'./temp/temp{i}.csv', sep=',', index=False)

                for model in models:
                    fut = executor.submit(run_model, i, model, timeList)
                    all_tasks.append(fut)

            total_tasks = len(all_tasks)
            finished_tasks = 0

            for future in as_completed(all_tasks):
                finished_tasks += 1
                time_index, model_name, f1_n = future.result()
                results_by_time.setdefault(time_index, {})[model_name] = f1_n
                print(f"Done: [{finished_tasks}/{total_tasks}] "
                      f"time {time_index} - model {model_name} - result: {f1_n}")

if __name__ == "__main__":
    from CMOS.semi.dataloader import load_Dataset_ellipticPlusPlus
    ellipticplusplus = load_Dataset_ellipticPlusPlus(data_timeList=None)
    test_get_predict(ellipticplusplus = ellipticplusplus)