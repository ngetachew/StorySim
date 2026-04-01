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

# Prompts 

values = [
    # 1
    "Read the following story and answer the question at the end. Note that all characters start in the hallway. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother. There is enough information to answer every question.",
    
    # 2
    "Please read the following story and answer the concluding question. Every character starts in the hallway. Characters who occupy the same location can observe each other's departures, while those in separate locations cannot. All questions are answerable, so be sure to provide an answer.",
    
    # 3
    """INSTRUCTIONS: Read the story and answer the question that follows.
    All characters begin in the hallway. When two characters share a space, they know where the other goes upon leaving; characters in different spaces lack this information. Every question has a sufficient basis in the story, and you must provide an answer.
    Story: """,
    
    # 4
    "Review the story and give the answer to the final question. Characters start out in the hallway. Characters in the same room always observe each other's movements, while characters in different rooms remain unaware. The necessary information is guarenteed to be available; provide the answer succinctly.",
    
    # 5
    "Read the following narrative and answer the question that appears at the end. All characters begin in the hallway. Shared locations allow characters to see one another move; separated characters have no visibility. Rely solely on the information in the story and give a direct answer.",
    
    # 6
    "Carefully read the story and respond to the final question. Every character starts in the hallway. Characters sharing a location know where others go when they leave; characters in different locations do not. The question can be answered from the text alone—state your conclusion briefly.",
    
    # 7
    """# Instructions
    Read the story and answer the question that follows.
    ### Rules
    - All characters begin in the hallway.  
    - When characters share a location, they observe each other's movements; those in different places cannot.  
    - The answer is fully determined by the story.  
    Provide a concise, final answer.""",
    
    # 8
    """# Instructions
    You will read a story involving characters moving between locations and then answer comprehension questions.
    ### Rules
    - All characters start in the hallway.
    - Characters who share a location can track each other's departures; characters who are apart cannot.
    - The question is completely answerable from the story.
    - Provide a single, final answer.""",
    
    # 9
    """Instructions: Read the story about characters moving between different locations, then answer the question that follows.
    Rules:
    - All characters begin in the hallway.
    - Characters in the same place can observe each other's departures; characters in different places cannot.
    - The question is fully answerable from the provided text.
    - Respond with one clear final answer.
    Story: """,
    
    # 10
    """You are an assistant tasked with reading a story about characters moving between locations and then answering questions based on it. Follow these rules:
    - All characters start in the hallway.
    - Characters who share a location can see where others go; characters who are apart lack this visibility.
    - The question is fully answerable from the narrative.
    - Provide one final answer.
    If it ever seems like information is missing, re-check the story—it always contains enough clues. Avoid repeating yourself.
    Story: """,

    # 11
    """You are a helpful reading assistant that will read the provided story about characters moving between locations, then answer comprehension questions at the end. Here are some rules that you must remember:

    - All of the characters start in the hallway. 
    - Characters who share a location know where the other goes when leaving; characters in separate locations have no visibility. 
    - The question is 100 percent answerable using the given information.
    - Output a single final answer.
    If you ever think there isn't enough information in the story, check again, because the questions are designed to be answerable by the prompt. Try not to repeat yourself.
    Story: """,

    # 12
    f"""INSTRUCTIONS: Read the following story and answer the question at the end. 
    Note that all characters start in the halllway. 
    If two characters are in the same location, then they know where eachother are.
    If one of them leaves the location, then the other will know where they went. 
    There is enough information to answer every question. You must provide an answer to every question. 
    
    Story: """,
]
print(len(values))





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
mislead = 30

models = [
              "deepseek-ai/DeepSeek-R1", 
              "gpt-4"]
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
print('Prompt test')

for v in range(len(values)):
    if v < 9:
        continue 
    print(f'Prompt {v}')
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
    mislead_distance = mislead

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
    intial_prompt = values[v]
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
        model_df.to_csv(f'~/scratch/heuristics-prompt-test/{model_name.replace("/","_")}_prompt{v}_test.csv')
    
for mn in model_knowns.keys():    
    for i in range(len(model_knowns[mn])):
        score = sum([1 for k in model_knowns[mn][i] if k == 'True'])
        print(f'Prompt: {values[i]}:\n {score}/{num_trials}, {len(model_unkowns[mn][i])} unkown')