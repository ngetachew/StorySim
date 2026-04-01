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
        except together.RateLimitError as e:
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
# Num of people
#values = [3,10,20,40]

# Mislead
values = [5,10, 20, 30, 40, 50, 60, 70, 80]
#values = [5, 30, 50, 80]
#values = [30,40]
#values = values[-2:]

#models = ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
models = ["Qwen/Qwen3.5-397B-A17B"]
# models = [    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
#               "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#               "deepseek-ai/DeepSeek-R1", 
#               "gpt-4"]
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
    graph = { 
        "room_1": ["room_2", "the_hallway","room_5"],
        "room_2": ["room_1", "room_3","the_hallway"],
        "room_3": ["room_2", "room_4","the_hallway"],
        "room_4": ["room_3", "room_5","room_1"],
        "room_5": ["room_4", "room_1","room_2"],
        "the_hallway": ["room_1", "room_4","room_2"]
    }


    locations = list(graph.keys())
    story_length = 100
    num_trials = 50
    
    mislead_distance = values[v]

    random.seed(25)

    for _ in range(num_trials):
        
        event_dict, max_actor = mislead_experiment(mislead_distance, story_length)
        storyboard = Storyboard('enters', graph, possible_people[:num_people], story_length, event_dict)

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
        d['Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
        df = pd.concat([df, pd.DataFrame(d)])   
    
    # Prompt model
    intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother. There is enough information to answer every question. Think step by step."
    tom_responses, wm_responses = [], []
    # Make a copy of each dataframe for each model
    dataframes = {model_name: df.copy() for model_name in models}

    # Helper: run async calls for non-OpenAI models and sync calls for OpenAI-style models
    def gather_responses_for_prompt(prompt_text, models_list):
        async_models = [m for m in models_list if ('gpt' not in m and 'o3' not in m)]
        sync_models = [m for m in models_list if m not in async_models]
        sync_models = models_list # TEMP: treat all models as async to avoid rate limits
        async_models = []

        responses = {}

        # Run async models in parallel
        if async_models:
            async def _run_async():
                coros = [run_llm_parallel(user_prompt=prompt_text, model=m) for m in async_models]
                return await asyncio.gather(*coros)

            try:
                async_results = asyncio.run(_run_async())
            except Exception as e:
                print(f'Error running async models: {e}')
                async_results = [''] * len(async_models)

            for m, r in zip(async_models, async_results):
                responses[m] = r

        # Run sync models one-by-one
        for m in sync_models:
            try:
                responses[m] = prompt_model(prompt_text, m)
            except Exception as e:
                print(f'Error calling model {m}: {e}')
                responses[m] = ''

        # Return responses in original order
        return [responses.get(m, '') for m in models_list]

    tom_responses = []
    for i, d in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f'Processing {i}/{len(df)} using models list')
        _ , row = d
        p = row['P1'].split(',')
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        prompt = f'{prompt_prefix}Q: Where does {p[0]} think {row["P2"]} is?\nA:'
        results = gather_responses_for_prompt(prompt, models)
        tom_responses.append(results)

    # For each model, fill dataframe column, compute scores, and save
    out_dir = os.path.expanduser('~/scratch/mislead-results')
    os.makedirs(out_dir, exist_ok=True)

    for idx, model_name in enumerate(models):
        model_df = dataframes[model_name]
        model_df['TOM Responses'] = [model_res[idx] for model_res in tom_responses]

        # Check scores
        outs = model_df.apply(lambda x: compute_score_unsure(x['Label'], x['TOM Responses']), axis=1)
        known = [k for k in outs if k == 'True' or k == 'False']
        model_knowns[model_name].append(known)
        unknown = [k for k in outs if k != 'True' and k != 'False']
        model_unkowns[model_name].append(unknown)

        print('UKNOWNS')
        for j in range(len(outs)):
            if outs.iloc[j] != 'True' and outs.iloc[j] != 'False':
                clean_out = outs.iloc[j].split('<think>')[-1]
                print(f'{clean_out}')
                print(f'Index {j}')
                print("\n-////==============////-\n")
        print(len(unknown))

        safe_name = model_name.replace('/', '_')
        model_df.to_csv(os.path.join(out_dir, f'{safe_name}_{values[v]}-mislead.csv'))
    
for mn in model_knowns.keys():    
    for i in range(len(model_knowns[mn])):
        score = sum([1 for k in model_knowns[mn][i] if k == 'True'])
    print(f'Model {mn}|  Mislead = {values[i]}: {score}/{num_trials}, {len(model_unkowns[mn][i])} unkown')