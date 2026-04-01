from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from together import Together
from experiment_defs import painting_experiment_2
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

transition_nodes = [
        "wander into the next room",
        "see a crowd in the next room and move over there",
        "double back to a previous room after realizing they missed something",
        "take a short break and then continue to the next room",
    ]

graph = {
    "enter the museum": [
        "notice a portrait on the wall and agree that it's well done",
        "see a portrait on the wall and conclude that it needs work",
        "inspect an impressionist painting and discuss its similarity to previous work",
        "look at an impressionist painting and discuss its originality",
        "smile with excitement as they recognize a famous work on the wall",
        "spend a while appreciating a painting of a landscape",
    ],

    # Opinion nodes — all lead to room transitions
    "notice a portrait on the wall and agree that it's well done": transition_nodes,
    "see a portrait on the wall and conclude that it needs work": transition_nodes,
    "inspect an impressionist painting and discuss its similarity to previous work": transition_nodes,
    "look at an impressionist painting and discuss its originality": transition_nodes,
    "smile with excitement as they recognize a famous work on the wall": transition_nodes,
    "spend a while appreciating a painting of a landscape":  transition_nodes,
    "notice a sculpture and agree it feels out of place": transition_nodes,
    "pause in front of an abstract piece and share their confusion":  transition_nodes,
    "admire a large mural and agree it dominates the room beautifully": transition_nodes,
    "glance at a watercolor and agree it feels unfinished":  transition_nodes,
    "study a still life and agree the composition is masterful":  transition_nodes,
    "are struck by a large oil painting and agree it is the best thing they have seen all day":  transition_nodes,
    "find a photograph that sparks a lengthy discussion about whether photography counts as art":  transition_nodes,
    "disagree briefly before landing on a shared opinion about a contemporary installation": transition_nodes,
    "linger in front of a medieval triptych and agree it has an eerie quality": transition_nodes,
    "discover a hidden gem in the corner and agree it was worth seeking out": transition_nodes,

    # Room transition nodes — all lead to the full set of opinion nodes
    "wander into the next room": [
        "notice a portrait on the wall and agree that it's well done",
        "see a portrait on the wall and conclude that it needs work",
        "inspect an impressionist painting and discuss its similarity to previous work",
        "look at an impressionist painting and discuss its originality",
        "smile with excitement as they recognize a famous work on the wall",
        "spend a while appreciating a painting of a landscape",
        "notice a sculpture and agree it feels out of place",
        "pause in front of an abstract piece and share their confusion",
        "admire a large mural and agree it dominates the room beautifully",
        "glance at a watercolor and agree it feels unfinished",
        "study a still life and agree the composition is masterful",
        "are struck by a large oil painting and agree it is the best thing they have seen all day",
        "find a photograph that sparks a lengthy discussion about whether photography counts as art",
        "disagree briefly before landing on a shared opinion about a contemporary installation",
        "linger in front of a medieval triptych and agree it has an eerie quality",
        "discover a hidden gem in the corner and agree it was worth seeking out",
    ],
    "see a crowd in the next room and move over there": [
        "notice a portrait on the wall and agree that it's well done",
        "see a portrait on the wall and conclude that it needs work",
        "inspect an impressionist painting and discuss its similarity to previous work",
        "look at an impressionist painting and discuss its originality",
        "smile with excitement as they recognize a famous work on the wall",
        "spend a while appreciating a painting of a landscape",
        "notice a sculpture and agree it feels out of place",
        "pause in front of an abstract piece and share their confusion",
        "admire a large mural and agree it dominates the room beautifully",
        "glance at a watercolor and agree it feels unfinished",
        "study a still life and agree the composition is masterful",
        "are struck by a large oil painting and agree it is the best thing they have seen all day",
        "find a photograph that sparks a lengthy discussion about whether photography counts as art",
        "disagree briefly before landing on a shared opinion about a contemporary installation",
        "linger in front of a medieval triptych and agree it has an eerie quality",
        "discover a hidden gem in the corner and agree it was worth seeking out",
    ],
    "double back to a previous room after realizing they missed something": [
        "notice a portrait on the wall and agree that it's well done",
        "see a portrait on the wall and conclude that it needs work",
        "inspect an impressionist painting and discuss its similarity to previous work",
        "look at an impressionist painting and discuss its originality",
        "smile with excitement as they recognize a famous work on the wall",
        "spend a while appreciating a painting of a landscape",
        "notice a sculpture and agree it feels out of place",
        "pause in front of an abstract piece and share their confusion",
        "admire a large mural and agree it dominates the room beautifully",
        "glance at a watercolor and agree it feels unfinished",
        "study a still life and agree the composition is masterful",
        "are struck by a large oil painting and agree it is the best thing they have seen all day",
        "find a photograph that sparks a lengthy discussion about whether photography counts as art",
        "disagree briefly before landing on a shared opinion about a contemporary installation",
        "linger in front of a medieval triptych and agree it has an eerie quality",
        "discover a hidden gem in the corner and agree it was worth seeking out",
    ],
    "take a short break and then continue to the next room": [
        "notice a portrait on the wall and agree that it's well done",
        "see a portrait on the wall and conclude that it needs work",
        "inspect an impressionist painting and discuss its similarity to previous work",
        "look at an impressionist painting and discuss its originality",
        "smile with excitement as they recognize a famous work on the wall",
        "spend a while appreciating a painting of a landscape",
        "notice a sculpture and agree it feels out of place",
        "pause in front of an abstract piece and share their confusion",
        "admire a large mural and agree it dominates the room beautifully",
        "glance at a watercolor and agree it feels unfinished",
        "study a still life and agree the composition is masterful",
        "are struck by a large oil painting and agree it is the best thing they have seen all day",
        "find a photograph that sparks a lengthy discussion about whether photography counts as art",
        "disagree briefly before landing on a shared opinion about a contemporary installation",
        "linger in front of a medieval triptych and agree it has an eerie quality",
        "discover a hidden gem in the corner and agree it was worth seeking out",
    ],
}


# Mislead
values = [5, 10, 20, 30, 40, 50, 60, 70, 80]
#values = [10, 20, 30, 40, 50, 60, 70, 80]

models = ["gpt-4"]

model_knowns = {m:[] for m in models}
model_unkowns = {m:[] for m in models}
possible_people = [
    "Alan", "Brian", "Charlie", "David", "Edward"
]


print('beginning')
print('Heurisitc: Num of people')

for v in range(len(values)):
    print(f'Mislead = {values[v]}')
    df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[], 'Last':[], 'CP_Loc':[]})
    num_people = 5


    locations = list(graph.keys())
    story_length = 10
    num_trials = 100

    random.seed(25)

    for _ in range(num_trials):
        
        event_dict, manual_actions_dict, max_actor = painting_experiment_2(story_length)
        storyboard = Storyboard('', graph, possible_people[:num_people], story_length, event_dict, manual_actions=manual_actions_dict)
        locations = list(graph.keys())

        sim = StorySimulator(
            people=["They", "The pair", "The two", "The friends", "The duo"],
            locations=locations,
            action="",
            manual_actions=manual_actions_dict,
            storyboard=storyboard,
            graph=graph
        )

        res = sim.run_simulation(story_length)
        story = sim.formal_to_story(res)

        d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
        d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
        d['P2'] = storyboard.actor_mapping[max_actor]
        d['Story'].append(story)
        d['Label'].append("Yes")
        df = pd.concat([df, pd.DataFrame(d)])   
    
    # Prompt model
    intial_prompt = f"Read the following story and answer the question at the end. In this story, two characters are visiting an art museum together. Read the story fully before answering the question."
    tom_responses = []
    
    # Make a copy of each dataframe for each model
    dataframes = {model_name: df.copy() for model_name in models}
    
    tom_total, wm_total = 0, 0
    for i, d in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f'Processing {i}/{len(df)} using {models}')
        _ ,row = d
        p = row['P1'].split(',')
        prompt_prefix = f"{intial_prompt}\n{row['Story']}.\n"
        prompt = f'{prompt_prefix}Q: How does {p[0]} think {row["P2"]} feels?\nA:'
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