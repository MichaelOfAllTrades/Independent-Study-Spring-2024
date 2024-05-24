import json
from datetime import datetime, timedelta

from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import pandas as pd


class GenerateAndCompareEmbeddings:
    def __init__(self, excel_file_name):
        self.excel_file_name = excel_file_name
        self.df = pd.read_excel(excel_file_name)

        self.all_nutrients = set()
        self.all_food_objects = {}

        # self.iterate_through_apis()

        self.all_food_embeddings = {}
        self.all_food_ingredients_embeddings = {}
        self.all_food_nutrition_embeddings = {}

        self.food1_embeddings = {}
        self.food2_embeddings = {}

        # self.create_ingredients_and_nutrition_embeddings_write_to_json(self)
        # self.create_ingredients_embeddings_write_to_json(self)
        # self.create_nutrition_embeddings_write_to_json(self)

    def get_food_data_from_api(self, food_name, api):
        food = None
        response = requests.get(f'{api}')

        if response.status_code == 200:
            data = response.json()
            if len(data['foods']) > 0:
                food = data['foods'][0]
                print(food_name, "found")
            else:
                print(food_name, "not found")
        else:
            print(f"Request failed with status code {response.status_code}")

        ingredients = food.get('ingredients').split(",")
        ingredients_list = [ingredient.strip() for ingredient in ingredients]
        serving_size_unit = food.get('servingSizeUnit')
        serving_size = food.get('servingSize')
        fraction = serving_size / 100

        nutrients = food.get('foodNutrients')

        current_nutrients = {}
        for nutrient in nutrients:
            if 'percentDailyValue' in nutrient:
                current_nutrients[nutrient['nutrientName']] = {
                    'value': nutrient['value'] * fraction,
                    'percentDailyValue': nutrient['percentDailyValue']
                }
            else:
                current_nutrients[nutrient['nutrientName']] = {
                    'value': nutrient['value'] * fraction,
                    'percentDailyValue': 0
                }

        # Update the global set of all nutrients
        self.all_nutrients.update(current_nutrients.keys())

        # Create a dictionary for all nutrients, filling in None for missing nutrients
        # all_nutrients_dict = {nutrient: current_nutrients.get(nutrient, None) for nutrient in all_nutrients}

        # Combine everything into a single food object
        food_object = {
            'name': food_name,
            'ingredients': ingredients_list,
            'serving_size_unit': serving_size_unit,
            'serving_size': serving_size,
            'fraction': fraction,
            'nutrients': current_nutrients
        }

        return food_object

    def iterate_through_apis(self):
        for index, row in self.df.iterrows():
            if pd.notna(row['fdc api url']):
                name = row['name']
                endpoint = row['fdc api url']
                self.all_food_objects[name] = self.get_food_data_from_api(name, endpoint)
        # for endpoint in all_endpoints:
        #     all_food_objects[endpoint['name']] = get_food_data_from_api(endpoint['name'], endpoint['api url'])
        #     # all_food_objects.append(get_food_data_from_api(endpoint['name'], endpoint['api url']))
        #     # print("type", type(all_food_objects[endpoint['name']]['nutrients']))
        for food_name, food_object in self.all_food_objects.items():
            print("type of food_object", type(food_object))
            current_nutrients = {}
            for nutrient in self.all_nutrients:
                if nutrient in food_object['nutrients']:
                    current_nutrients[nutrient] = food_object['nutrients'][nutrient]
                else:
                    current_nutrients[nutrient] = {
                        'value': 0,
                        'percentDailyValue': 0
                    }
            food_object['nutrients'] = current_nutrients

        # for food_name, food_object in all_food_objects.items():
        #     print(food_object['nutrients'])

        # put all_food_objects into a JSON file called 'official data 1.json'
        with open('official data 1.json', 'w') as f:
            json.dump(self.all_food_objects, f, indent=4)

    def create_ingredients_and_nutrition_embeddings_write_to_json(self):
        # get the ingredients and nutrition data for the food from json file
        all_food_data = {}
        with open('official data 1.json', 'r') as f:
            all_food_data = json.load(f)

        for food_name, food_data in all_food_data.items():
            food_data = all_food_data[food_name]
            ingredients = food_data['ingredients']
            nutrition_data = [[nutrient['value'], nutrient['percentDailyValue']] for nutrient in
                              food_data['nutrients'].values()]

            # Train the Word2Vec model on our ingredients
            model = Word2Vec([ingredients], min_count=1)

            # Get the ingredient embeddings
            ingredients_embeddings = np.mean([model.wv[ingredient] for ingredient in ingredients], axis=0)

            # Normalize the nutrition values to have the same scale as the embeddings
            scaler = StandardScaler()
            nutrition_scaled = scaler.fit_transform(nutrition_data)

            # Combine the embeddings
            combined_embeddings = np.concatenate((ingredients_embeddings, nutrition_scaled.flatten()))

            self.all_food_embeddings[food_name] = {
                "name": food_name,
                "embeddings": combined_embeddings.tolist()
            }
            print(food_name, "done")

        with open('all food embeddings.json', 'w') as f:
            json.dump(self.all_food_embeddings, f, indent=4)

    def create_ingredients_embeddings_write_to_json(self):
        # get the ingredients data for the food from json file
        all_food_data = {}

        with open('official data 1.json', 'r') as f:
            all_food_data = json.load(f)

        for food_name, food_data in all_food_data.items():
            food_data = all_food_data[food_name]
            ingredients = food_data['ingredients']

            # Train the Word2Vec model on our ingredients
            model = Word2Vec([ingredients], min_count=1)

            # Get the ingredient embeddings
            ingredients_embeddings = np.mean([model.wv[ingredient] for ingredient in ingredients], axis=0)

            self.all_food_ingredients_embeddings[food_name] = {
                "name": food_name,
                "embeddings": ingredients_embeddings.tolist()
            }
            print(food_name, "done")

        with open('all food ingredients embeddings.json', 'w') as f:
            json.dump(self.all_food_ingredients_embeddings, f, indent=4)

    def create_nutrition_embeddings_write_to_json(self):
        # get the nutrition data for the food from json file
        all_food_data = {}

        with open('official data 1.json', 'r') as f:
            all_food_data = json.load(f)

        for food_name, food_data in all_food_data.items():
            food_data = all_food_data[food_name]
            nutrition_data = [[nutrient['value'], nutrient['percentDailyValue']] for nutrient in
                              food_data['nutrients'].values()]

            # Normalize the nutrition values to have the same scale as the embeddings
            scaler = StandardScaler()
            nutrition_scaled = scaler.fit_transform(nutrition_data)

            self.all_food_nutrition_embeddings[food_name] = {
                "name": food_name,
                "embeddings": nutrition_scaled.flatten().tolist()
            }
            print(food_name, "done")

        with open('all food nutrition embeddings.json', 'w') as f:
            json.dump(self.all_food_nutrition_embeddings, f, indent=4)

    def compare_foods_by_both(self, food1, food2):
        # open the JSON file with the food data
        with open('all food embeddings.json', 'r') as f:
            all_embeddings = json.load(f)
            self.food1_embeddings = all_embeddings[food1]
            self.food2_embeddings = all_embeddings[food2]

        # Calculate the cosine similarity between the combined embeddings
        similarity = cosine_similarity([self.food1_embeddings['embeddings']], [self.food2_embeddings['embeddings']])

        # Normalize to 0-1 range
        normalized_similarity = (similarity + 1) / 2

        # Convert to percentage
        percentage_similarity = normalized_similarity * 100

        print(
            f"The 'ingredient AND nutritional' similarity between '{food1}' and '{food2}' is {percentage_similarity[0][0]:.2f}%")

    def compare_foods_by_ingredients(self, food1, food2):
        # open the JSON file with the food data
        with open('all food ingredients embeddings.json', 'r') as f:
            all_embeddings = json.load(f)
            self.food1_embeddings = all_embeddings[food1]
            food2_embeddings = all_embeddings[food2]

        # Calculate the cosine similarity between the combined embeddings
        similarity = cosine_similarity([self.food1_embeddings['embeddings']], [self.food2_embeddings['embeddings']])

        # Normalize to 0-1 range
        normalized_similarity = (similarity + 1) / 2

        # Convert to percentage
        percentage_similarity = normalized_similarity * 100

        print(f"The 'ingredients' similarity between '{food1}' and '{food2}' is {percentage_similarity[0][0]:.2f}%")

    def compare_foods_by_nutrition(self, food1, food2):
        # open the JSON file with the food data
        with open('all food nutrition embeddings.json', 'r') as f:
            all_embeddings = json.load(f)
            self.food1_embeddings = all_embeddings[food1]
            self.food2_embeddings = all_embeddings[food2]

        # Calculate the cosine similarity between the combined embeddings
        similarity = cosine_similarity([self.food1_embeddings['embeddings']], [self.food2_embeddings['embeddings']])

        # Normalize to 0-1 range
        normalized_similarity = (similarity + 1) / 2

        # Convert to percentage
        percentage_similarity = normalized_similarity * 100

        print(f"The 'nutritional' similarity between '{food1}' and '{food2}' is {percentage_similarity[0][0]:.2f}%")
