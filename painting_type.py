import pandas as pd
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import painting_experiment_2, painting_dislike

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

# def compute_score_unsure(label, response, starting_loc='hallway'):
#     #label = label.split(',')[0][2:-1]
#     base = f'{label.split("_")[0]}_' if not label.startswith(starting_loc) else label
#     response = response.split("\n")[-1]
#     response = response.lower().replace("_",' ')
#     if response.count(base) <= 1:
#         return str(label in response or label.replace('_'," ") in response or 
#                    label.replace('_',"") in response or
#                    (starting_loc
#                     in label and starting_loc in response) or
#                    label.replace(" ",'') in response)
#     elif 'Therefore,' in response:
#         response = response.split('Therefore,')[-1]
#         return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or (starting_loc in label and starting_loc in response))
#     return f'{response}, {label}'

def compute_score_unsure(label, response, starting_loc='hallway'):
    #label = label.split(',')[0][2:-1]
    base = label.lower()
    response = response.split("\n")[-1] if "\n" in response else response
    response = response.lower().replace("_",' ')
    if response.count(base) <= 1:
        return str(base in response)
    elif 'Therefore,' in response:
        response = response.split('Therefore,')[-1]
        return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or (starting_loc in label and starting_loc in response))
    return f'{response}, {label}'

graph = {
    "the impressionist painting": ["the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the abstract painting": ["the impressionist painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the minimalist painting": ["the impressionist painting", "the abstract painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the landscape painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the portrait painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the surrealist painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the watercolor painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the oil painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the still life painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the fresco": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the mural": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the sketch": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the lithograph": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the expressionist painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the cubist painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the baroque painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the renaissance painting", "the pop art piece", "the street art piece", "the triptych"],
    "the renaissance painting": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the pop art piece", "the street art piece", "the triptych"],
    "the pop art piece": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the street art piece", "the triptych"],
    "the street art piece": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the triptych"],
    "the triptych": ["the impressionist painting", "the abstract painting", "the minimalist painting", "the landscape painting", "the portrait painting", "the surrealist painting", "the watercolor painting", "the oil painting", "the still life painting", "the fresco", "the mural", "the sketch", "the lithograph", "the expressionist painting", "the cubist painting", "the baroque painting", "the renaissance painting", "the pop art piece", "the street art piece"],
        "start": [
        "the impressionist painting",
        "the abstract painting",
        "the minimalist painting",
        "the landscape painting",
        "the portrait painting",
        "the surrealist painting",
        "the watercolor painting",
        "the oil painting",
        "the still life painting",
        "the fresco",
        "the mural",
        "the sketch",
        "the lithograph",
        "the expressionist painting",
        "the cubist painting",
        "the baroque painting",
        "the renaissance painting",
        "the pop art piece",
        "the street art piece",
        "the triptych",
    ]
}

possible_people = [
    "Alan", "Brian", "Charlie", "David", "Edward", "Frank", "George", "Henry", "Ian", "James", "Kevin", "Liam", "Michael", "Nathan", "Oliver", "Peter", "Quentin", "Robert", "Samuel", "Thomas", "Phil", "Victor", "William", "Xavier", "Yann", "Zachary"
]

df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[], 'Last':[], 'CP_Loc':[]})
num_people = len(possible_people)


locations = list(graph.keys())
story_length = 15
num_trials = 50

random.seed(25)

for _ in range(num_trials):
    
    event_dict, manual_actions_dict, max_actor, max_loc = painting_dislike(story_length)
    relations = ['looks at', 'ponders', 'checks out', "carefully inspects", "views"]
    storyboard = Storyboard(relations[0], graph, possible_people[:num_people], story_length, event_dict, manual_actions=manual_actions_dict)
    locations = list(graph.keys())

    sim = StorySimulator(
        people=possible_people,
        locations=locations,
        action=relations,
        manual_actions=manual_actions_dict,
        storyboard=storyboard,
        graph=graph
    )

    res = sim.run_simulation(story_length)
    story = sim.formal_to_story(res)

    d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
    #r = random.choice([p for p in possible_people if p in story and p != storyboard.actor_mapping["0"] and p != storyboard.actor_mapping["1"]])
    d['P1'] = ','.join(list(reversed([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])))
    d['P2'] = storyboard.actor_mapping[max_actor]
    d['Story'].append(story)
    #label = storyboard.loc_mapping[max_loc]
    label = storyboard.actor_mapping["0"]
    d['Label'].append(label)
    df = pd.concat([df, pd.DataFrame(d)])  


'''
-- Prompt model --
'''
intial_prompt = f"Read the following story and answer the question at the end. In this story, characters are visiting an art gallery and looking at different paintings on the wall. Characters looking at the same painting are in the same room as eachother and can freely observe one another's actions and speak with eachother. The story is guarenteed to have enough information to answer the question, so read it carefully."
# Add CoT
#intial_prompt += " Think step by step before answering the question."

models = ["gpt-4", "o3-mini","deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]
models = models[1:]

# Directory for outputs
scratch_dir = os.path.expanduser('~/scratch')
out_dir = os.path.join(scratch_dir, 'painting2')
os.makedirs(out_dir, exist_ok=True)

for model in models:
    print(f'\nRunning model: {model}')
    responses = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f'  Processing {i}/{len(df)} for {model}')
        p = row['P1'].split(',')
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        prompt = f'{prompt_prefix}Q: From {p[0]}\'s perspective, who likely doesn\'t enjoy paintings?\nA:'
        try:
            response = prompt_model(prompt, model)
        except Exception as e:
            print(f'    Error calling model {model} at index {i}: {e}')
            response = ''
        responses.append(response)

    model_df = df.copy(deep=True)
    model_df['Response'] = responses
    model_df['Score'] = model_df.apply(lambda r: compute_score_unsure(r['Label'], r['Response']), axis=1)

    correct = sum(model_df['Score'] == 'True')
    total = len(model_df)
    accuracy = (correct / total) * 100 if total else 0
    print(f'Accuracy for {model}: {correct}/{total} ({accuracy:.1f}%)')

    safe_model = model.replace('/', '_').replace(':', '_')
    output_path = os.path.join(out_dir, f'painting_type_results_{safe_model}.csv')
    model_df.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')

