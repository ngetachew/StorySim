#from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import mislead_experiment, second_order_tom_experiment
import asyncio
import together
from together import AsyncTogether, Together

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')

client = Together()


async def wait_for_res(coros):
    results = await asyncio.gater(*coros)
    return results

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

async def run_llm_parallel(user_prompt : str, model : str, system_prompt : str = None):
    """Run parallel LLM call with a reference model."""
    async_client = AsyncTogether()
    for sleep_time in [1, 2, 4]:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
    
            messages.append({"role": "user", "content": user_prompt})

            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            break
        except together.error.RateLimitError as e:
            print(e)
            await asyncio.sleep(sleep_time)
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
graph = {
    # Start node – all characters start here
    "your_phone": [
        "doctors_office",
        "boss",
        "school_office",
        "bank",
        "city_hall",
        "landlord",
    ],

    # Medical / health related
    "doctors_office": ["your_phone", "bank", "city_hall"],

    # Work / professional
    "boss": ["your_phone", "bank", "school_office"],

    # Education
    "school_office": ["your_phone", "boss", "city_hall"],

    # Money / services
    "bank": ["your_phone", "doctors_office", "boss", "city_hall"],

    # Government / authority
    "city_hall": ["your_phone", "doctors_office", "school_office", "bank", "landlord"],

    # Housing
    "landlord": ["your_phone", "city_hall"],
}



# Mislead
values = [5, 10, 20, 30, 40, 50, 60, 70, 80]
#values = [10, 20, 30, 40, 50, 60, 70, 80]

models = ["gpt-4"]

model_knowns = {m:[] for m in models}
model_unkowns = {m:[] for m in models}
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
print('Heurisitc: Num of people')

for v in range(len(values)):
    print(f'Mislead = {values[v]}')
    df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[], 'Last':[], 'CP_Loc':[]})
    num_people = 7


    locations = list(graph.keys())
    story_length = 100
    num_trials = 100
    mislead_distance = values[v]

    random.seed(25)

    for _ in range(num_trials):
        
        event_dict, max_actor = mislead_experiment(mislead_distance, story_length)
        storyboard = Storyboard('joins_a_call_with_the', graph, possible_people[:num_people], story_length, event_dict)
        locations = list(graph.keys())

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            action="joins_a_call_with_the",
            storyboard=storyboard,
            graph=graph
        )

        res = sim.run_simulation(story_length)
        story = sim.formal_to_story(res)
        print(story)
        

        d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
        d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
        d['P2'] = storyboard.actor_mapping[max_actor]
        d['Story'].append(story)
        d['Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
        df = pd.concat([df, pd.DataFrame(d)])   
    
    # Prompt model
    # TODO: Go over this again
    intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same conference call can tell where eachother go when someone hangs up. If characters are in different calls, they don't know which call another person is in. Assume characters remain in a call until you find out that they're in a different one. There is enough information to answer every question."
    tom_responses, wm_responses = [], []
    
    # Make a copy of each dataframe for each model
    dataframes = {model_name: df.copy() for model_name in models}
    
    tom_total, wm_total = 0, 0
    for i, d in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f'Processing {i}/{len(df)} using {models}')
        _ ,row = d
        p = row['P1'].split(',')
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        #prompt = f'{prompt_prefix}Q: Where does {p[0]} think {p[1]} thinks {row["P2"]} is?\nA:'
        prompt = f'{prompt_prefix}Q: Where does {p[0]} think {row["P2"]} is?\nA:'
        #print(prompt)
        async def _gather_tasks():
            coros = [run_llm_parallel(user_prompt=prompt, model=model) for model in models[:-1]]
            return await asyncio.gather(*coros)

        # Run it without a main() wrapper
        results = asyncio.run(_gather_tasks())
        # Hardcode gpt4
        results.append(prompt_model(prompt, models[-1]))  
        tom_responses.append(results)
    for idx, model_name in enumerate(models):
        model_df = dataframes[model_name]
        model_df['TOM Responses'] = [model_res[idx] for model_res in tom_responses ]
        # Check scores
        outs= model_df.apply(lambda x: compute_score_unsure(x['Label'], x['TOM Responses']), axis=1)
        known = [k for k in outs if k == 'True' or k == 'False']
        model_knowns[model_name].append(known)
        unknown = [k for k in outs if k != 'True' and k != 'False']
        model_unkowns[model_name].append(unknown)
        print('UKNOWNS')
        for i in range(len(outs)):
            if outs.iloc[i] != 'True' and outs.iloc[i] != 'False':
                clean_out = outs.iloc[i].split('<think>')[-1]
                print(f'{clean_out}')
                print(f'Index {i}')
                print("\n-////==============////-\n")
        print(len(unknown))
        model_df.to_csv(f'~/scratch/conference-call-story/{model_name.replace("/","_")}_{values[v]}-conference-fixed.csv')
    
for mn in model_knowns.keys():    
    for i in range(len(model_knowns[mn])):
        score = sum([1 for k in model_knowns[mn][i] if k == 'True'])
        print(f'Mislead = {values[i]}: {score}/{num_trials}, {len(model_unkowns[mn][i])} unkown')