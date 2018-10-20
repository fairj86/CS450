import pandas as pd
import numpy as np


autism = pd.read_csv('C://Users/jfair/Desktop/prove3/autism.csv')
car = pd.read_csv('C://Users/jfair/Desktop/prove3/Car.csv')
mpg = pd.read_csv('C://Users/jfair/Desktop/prove3/mpg.csv')

#cleaning up Autism Dataset
autism = autism.replace("?", np.nan)
autism = autism.dropna()

def handle_non_numerical_data(autism):
    columns = autism.columns.values

    for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if autism[column].dtype != np.int64 and autism[column].dtype !=np.float64:
                column_contents = autism[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1
    return autism

# cleaning up Car Dataset
car = car.replace("?", np.nan)
car = car.dropna()


def handle_non_numerical_data(car):
    columns = car.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if car[column].dtype != np.int64 and car[column].dtype != np.float64:
            column_contents = car[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
    return car

# cleaning up MPG Dataset
mpg = mpg.replace("?", np.nan)
mpg = mpg.dropna()


def handle_non_numerical_data(mpg):
    columns = mpg.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if mpg[column].dtype != np.int64 and mpg[column].dtype != np.float64:
            column_contents = mpg[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
    return mpg

def k_nearest_neighbors_mpg(mpg, predict, k=3):
    dist = []
    for group in mpg:
        for features in mpg[group]:
            eucld_dist = np.sqrt(np.linalg.norm(np.array(features)-np.array(predict)))
            dist.append([eucld_dist, group])

    mpg_avg = [i[1] for i in sorted(dist)[:k]]
    mpg_results = Counter(mpg_avg).most_common(1)[0][0]
    return mpg_results



def k_nearest_neighbors_autism(autism, predict, k=3):
    dist = []
    for group in autism:
        for features in autism[group]:
            eucld_dist = np.sqrt(np.linalg.norm(np.array(features) - np.array(predict)))
            dist.append([eucld_dist, group])

    autism_avg = [i[1] for i in sorted(dist)[:k]]
    autism_results = Counter(autism_avg).most_common(1)[0][0]
    return autism_results


def k_nearest_neighbors_car(car, predict, k=3):
    dist = []
    for group in car:
        for features in car[group]:
            eucld_dist = np.sqrt(np.linalg.norm(np.array(features) - np.array(predict)))
            dist.append([eucld_dist, group])

    car_avg = [i[1] for i in sorted(dist)[:k]]
    car_results = Counter(car_avg).most_common(1)[0][0]
    return car_results


#print("Accuracy Car Results : ", k_nearest_neighbors_car())
#print("Accuracy Mpg Results : ", mpg_results)
#print("Accuracy Autism Results : ", autism_results)
