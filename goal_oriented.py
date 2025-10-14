from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import goal_oriented

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')


def prompt_model(prompt, model):
    if 'gpt' in model or 'o3' in model:
        client = OpenAI()
    else:
        client = Together()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content

def compute_score_unsure(label, response, starting_loc='hallway'):
    #label = label.split(',')[0][2:-1]
    base = f'{label.split("_")[0]}_' if not label.startswith(starting_loc) else label
    response = response.split("\n")[-1]
    response = response.lower().replace("_",' ')
    if response.count(base) <= 1:
        return str(label in response or label.replace('_'," ") in response or 
                   label.replace('_',"") in response or
                   (starting_loc
                    in label and starting_loc in response) or
                   label.replace(" ",'') in response)
    elif 'Therefore,' in response:
        response = response.split('Therefore,')[-1]
        return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or (starting_loc in label and starting_loc in response))
    return f'{response}, {label}'



'''
Actual Experiements
'''
# Num of people
#values = [3,10,20,40]

# Mislead
values = [5,10, 20, 30, 40, 50, 60, 70, 80]
values = [20, 30]
values = [5]
#values = [30,40]
#values = values[-2:]

knowns, unkowns = [], []
possible_people = [
    "Alice", "Bob", "Charlie", "Danny", "Edward",
    "Frank", "Georgia", "Hank", "Isaac", "Jake",
    "Kevin", "Liam", "Mia", "Nina", "Oliver",
    "Paula", "Quinn", "Rachel", "Steve", "Tina",
    "Uma", "Victor", "Wendy", "Xander", "Yara",
    "Zane", "Amber", "Brandon", "Carmen", "Derek",
    "Elena", "Felix", "Grace", "Harvey", "Ivy",
    "Jasmine", "Kyle", "Leah", "Miles", "Naomi",
    "Henry", "Wyatt", "Jose", "Neil","Seth"
]


print('beginning')

df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[], 'Last':[], 'CP_Loc':[]})
num_people = 12
graph = { 
    "room_1": ["room_2", "the_hallway","room_5"],
    "room_2": ["room_1", "room_3","the_hallway"],
    "room_3": ["room_2", "room_4","the_hallway"],
    "room_4": ["room_3", "room_5","room_1"],
    "room_5": ["room_4", "room_1","room_2"],
    "the_hallway": ["room_1", "room_4","room_2"]
}


locations = list(graph.keys())
story_length = 25
num_trials = 1


random.seed(25)

for _ in range(num_trials):
    
    event_dict, manual_actions, max_actor = goal_oriented(story_length)
    storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict, manual_actions=manual_actions)

    sim = StorySimulator(
        people=possible_people[:num_people],
        locations=locations,
        action="enters",
        storyboard=storyboard,
        graph=graph
    )

    res = sim.run_simulation(story_length)

    story = sim.formal_to_story(res)
    d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
    d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
    d['P2'] = storyboard.actor_mapping[max_actor]
    d['Story'].append(story)
    d['Label'].append('')
    df = pd.concat([df, pd.DataFrame(d)])
    print(story)

#     # Prompt model
#     intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother."
#     tom_responses, wm_responses = [], []
#     #model_choice =  "deepseek-ai/DeepSeek-R1"
#     model_choice =  "o3-mini"
#     #model_choice = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#     tom_total, wm_total = 0, 0
#     for i, d in enumerate(df.iterrows()):
#         if i % 10 == 0:
#             print(f'Processing {i}/{len(df)}')
#         _ ,row = d
#         p = row['P1'].split(',')[-1]
#         prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
#         prompt = f'{prompt_prefix}Q: Where does {p} think {row["P2"]} is?\nA:'
#         answer = prompt_model(prompt, model_choice)
#         tom_responses.append(answer)
#     df['TOM Responses'] = tom_responses

#     outs= df.apply(lambda x: compute_score_unsure(x['Label'], x['TOM Responses']), axis=1)
#     known = [k for k in outs if k == 'True' or k == 'False']
#     knowns.append(known)
#     unknown = [k for k in outs if k != 'True' and k != 'False']
#     unkowns.append(unknown)
#     print('UKNOWNS')
#     for i in range(len(outs)):
#         if outs.iloc[i] != 'True' and outs.iloc[i] != 'False':
#             clean_out = outs.iloc[i].split('<think>')[-1]
#             print(f'{clean_out}')
#             print(f'Index {i}')
#             print("\n-////==============////-\n")
#     print(len(unknown))
#     df.to_csv(f'new_mislead/{model_choice.replace("/","_")}_{values[v]}.csv')
        
# for i in range(len(knowns)):
#     score = sum([1 for k in knowns[i] if k == 'True'])
#     print(f'Mislead = {values[i]}: {score}/{num_trials}, {len(unkowns[i])} unkown')
        
        
    