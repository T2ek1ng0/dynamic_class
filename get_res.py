import pickle
import numpy as np


def read_sgbest_from_pickle(filename="test_results.pkl"):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if 'avg_dist' in data:
            sgbest_data = data['avg_dist']

            for problem_name, result_dict in sgbest_data.items():
                print(problem_name)

                for algo_name, list_of_res in result_dict.items():
                    print(f"agent: {algo_name}")

                    if not list_of_res:
                        print("(无数据)")
                        continue

                    print(f"mean={np.mean(list_of_res)}, std={np.std(list_of_res)}")

        else:
            print(f"错误：在文件 '{filename}' 中未找到 'avg_dist' 键。")

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在。请确保文件在正确的路径下。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")


if __name__ == "__main__":
    pkl_file = r"output\test\20251102T154506_dynamic-problem_easy\test_results.pkl"
    read_sgbest_from_pickle(pkl_file)
