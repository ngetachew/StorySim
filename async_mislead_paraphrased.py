#from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import mislead_experiment
import asyncio
import together
from together import AsyncTogether, Together

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_KEY')

client = Together()

PARAPHRASE_PROMPT = """You are a helpful paraphrasing assistant. Please paraphrase the following two sentences so that it sounds less template-like. Do not change the meaning of the sentence, and do not paraphrase or change the name of any locations that have and underscore("_"). For example, city_hall can become city hall, but not "town hall". Try not to make your paraphrased version much longer than the original input.

Sentence: {input_sentence}"""

def list_to_pairs_loop(data_list):
    """
    Converts a list into a list of pairs using a for loop.
    
    If the list length is odd, the last element is placed in a partial tuple.
    """
    pairs = []
    # Iterate with a step of 2 over the list's indices
    for i in range(0, len(data_list), 2):
        # Check if a second element exists
        if i + 1 < len(data_list):
            pairs.append((data_list[i], data_list[i+1]))
        else:
            pairs.append((data_list[i],)) # Append the single leftover element as a tuple
    return pairs


def llm_paraphrase(events):
    story = []
    new_events = ['. '.join(e) if len(e) == 2 else e[0] for e in list_to_pairs_loop(events)]
    for e in new_events:
        paraphrased = prompt_model(PARAPHRASE_PROMPT.format(input_sentence=e), 'gpt-4o-mini', temperature=1.3)
        story.append(paraphrased)
    return story


async def wait_for_res(coros):
    results = await asyncio.gater(*coros)
    return results

def prompt_model(prompt, model, temperature=1):
    if 'gpt' in model or 'o3' in model or 'mini' in model:
        client = OpenAI()
    else:
        client = Together()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
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
# Num of people
#values = [3,10,20,40]

# Mislead
values = [5, 10, 20, 30, 40, 50, 60, 70, 80]
values = [10, 20, 50, 60, 80]
#values = values[-2:]

#models = ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
# models = [    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
#               "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#               "Qwen/QwQ-32B-Preview",
#               "deepseek-ai/DeepSeek-R1", 
#               "gpt-4"]
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


    locations = list(graph.keys())
    story_length = 100
    num_trials = 100
    mislead_distance = values[v]

    random.seed(25)

    for k in range(num_trials):
        if k % 5 == 0:
            print(f'{k} out of {num_trials} summarized')

        event_dict, max_actor = mislead_experiment(mislead_distance, story_length)
        storyboard = Storyboard('joins_a_call_with_the', graph, possible_people[:num_people], story_length, event_dict)

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            action="joins_a_call_with_the",
            storyboard=storyboard,
            graph=graph
        )

        res = sim.run_simulation(story_length)

        story = sim.formal_to_story(res)
        story = ''.join(llm_paraphrase(story.split('. ')))
        d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
        d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
        d['P2'] = storyboard.actor_mapping[max_actor]
        d['Story'].append(story)
        d['Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
        df = pd.concat([df, pd.DataFrame(d)])   
    
    # Prompt model
    intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start on their phone(your_phone). Characters in the same conference call can tell where eachother go when someone hangs up. If characters are in different calls, they don't know which call another person is in. Assume characters remain in a call until you find out that they're in a different one. There is enough information to answer every question."
    tom_responses, wm_responses = [], []
    
    # Make a copy of each dataframe for each model
    dataframes = {model_name: df.copy() for model_name in models}
    
    tom_total, wm_total = 0, 0
    for i, d in enumerate(df.iterrows()):
        if i % 5 == 0:
            print(f'Processing {i}/{len(df)} using {models}')
        _ ,row = d
        p = row['P1'].split(',')
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        #prompt = f'{prompt_prefix}Q: Where does {p[0]} think {p[1]} thinks {row["P2"]} is?\nA:'
        prompt = f'{prompt_prefix}Q: Where does {p[0]} think {row["P2"]} is?\nA:'
        #print(prompt)
        # async def _gather_tasks():
        #     coros = [run_llm_parallel(user_prompt=prompt, model=model) for model in models[:-1]]
        #     return await asyncio.gather(*coros)

        # Run it without a main() wrapper
        # results = asyncio.run(_gather_tasks())
        # Hardcode gpt4
        # results.append(prompt_model(prompt, models[-1]))  
        results = [prompt_model(prompt, models[0])]
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
        model_df.to_csv(f'~/scratch/heuristics-paraphrase-test/{model_name.replace("/","_")}_{values[v]}-confcall-test.csv')
    
for mn in model_knowns.keys():    
    for i in range(len(model_knowns[mn])):
        score = sum([1 for k in model_knowns[mn][i] if k == 'True'])
        print(f'Mislead = {values[i]}: {score}/{num_trials}, {len(model_unkowns[mn][i])} unkown')
