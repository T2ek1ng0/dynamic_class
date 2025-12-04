import pickle
import numpy as np


def get_result(filename="test_results.pkl"):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        dist_data = data.get('avg_dist')
        time_data = data.get('T2')
        assert dist_data is not None, f'{filename} 中不存在 avg_dist 键'
        assert time_data is not None, f'{filename} 中不存在 T2 键'
        for problem_name in dist_data.keys():
            dist_dict = dist_data[problem_name]
            time_dict = time_data[problem_name]
            print(problem_name)
            for key in dist_dict:
                dist_list = np.array(dist_dict[key])
                time_list = np.array(time_dict[key])
                time_list /= 1000
                print(f'{key}: avg_dist: {np.mean(dist_list):.2f}({np.std(dist_list):.2f}), time: {np.mean(time_list):.2f}({np.std(time_list):.2f}) second/per problem')

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")


if __name__ == "__main__":
    pkl_file = "file path to your test_results.pkl"
    get_result(pkl_file)
