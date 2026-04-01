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

PROMPT_TEMPLATE = """You are a helpful story generating assistant. You task is to generate a story according to the following specifications. Only output the story and nothing else.
Specifications:
1. The story must contain 100 events.
2. Each event must be 1 sentence long ending with a period.
3. The event will always be a character entering a new location from one of the possible locations. Write the event as 'PERSON enters LOCATION'
4. The story must contain the following required events

REQUIRED EVENTS:
- By event number 10, two characters must be in the same location
- In time step 11, one of these characters must go to a second location
- In time step {MISLEAD_STEP}, this character must go to a third location
- These two characters should not make any moves other than what's specified above

POSSIBLE LOCATIONS: {LOCATION_LIST}
CHARACTER NAMES: {CHAR_NAMES}"""


def count_non_single_capitalized(sentences: list[str]) -> int:
    count = 0
    for s in sentences:
        words = s.split()
        # Count words whose first character is uppercase
        num_capitalized = sum(1 for w in words if w[0].isupper())
        if num_capitalized != 1:
            count += 1
    return count


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
# Num of people
#values = [3,10,20,40]

# Mislead
#values = [5,30,70]
values = [50]
#values = values[-2:]

#models = ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
# models = [    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
#               "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#               "Qwen/QwQ-32B-Preview",
#               "deepseek-ai/DeepSeek-R1", 
#               "gpt-4"]a
models = ["gpt-4"]

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
for mn in models:
    print(f'MODEL: {mn}')
    for v in range(len(values)):
        model_knowns = {m:[] for m in models}
        print(f'Mislead = {values[v]}')
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
        num_trials = 30
        mislead_distance = values[v]

        random.seed(25)

        for x in range(num_trials):
            if x % 10 == 0:
                print(f'{x} / {num_trials}')
            # Ask an LLM to generate a story based on the storyboard
            char_list = random.sample(possible_people, num_people)
            location_list = list(graph.keys())
            prompt = PROMPT_TEMPLATE.format(CHAR_NAMES=char_list, LOCATION_LIST=location_list, MISLEAD_STEP=10+values[v])
            # Prompt LLM
            response = prompt_model(prompt, mn)
            model_knowns[mn].append(response)

        model_res = model_knowns[mn]
    
        actor_a = ''
        actor_b = ''
        length_misses = []
        mislead_misses = []
        name_misses = []
        total = 0
        for idx, res in enumerate(model_res):
            
            events = [r.strip() for r in res.split('.') if not r.strip().isnumeric()]
            
            # Check for length
            if len(events) != 100:
                length_misses.append(idx)

            # Check for correct storyboard events
            actor_a = events[9].split(' ')[0]
            actor_b = events[10].split(' ')[0]
            n_values = len(values)
            
            if 10+values[v] >= len(events) or  not (actor_b in events[10] and actor_b in events[10+values[v]]):
                mislead_misses.append(idx)
            
            # Check for correct people
            if count_non_single_capitalized(events) > 0:
                name_misses.append(idx)
            
        total += len(set(length_misses).intersection(set(mislead_misses)).intersection(set(name_misses)))
                
        print(f'---- MISLEAD {values[v]} results ----')        
        print(f'Number of length mistakes for {mn}: {len(length_misses)}')
        print(f'Number of mislead misses for {mn}: {len(mislead_misses)}')
        print(f'Number of name misses for {mn}: {len(name_misses)}')
        print(f'===== Total wrong for {mn}: {total} =====')

        df = pd.DataFrame(model_knowns)
        df.to_csv(f'~/scratch/llm-test/llm-test-results-{values[v]}.csv')

