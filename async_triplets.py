#from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AsyncOpenAI, OpenAI
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
async_client = AsyncTogether()
parallelized = False

async def wait_for_res(coros):
    results = await asyncio.gater(*coros)
    return results

async def prompt_model(prompt, model):
    if 'gpt' in model or 'o3' in model:
        client = AsyncOpenAI()
    else:
        client = Together()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content

def prompt_model_synchronous(prompt, model):
    try:
        if 'gpt' in model or 'o3' in model:
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
        else:
            client = Together()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000
            )
        #print(f'Response for {model}: {response}')
        return response.choices[0].message.content
    except Exception as e:
        print(f'Error calling model {model}: {e}')
        return ''

async def run_llm_parallel(user_prompts : list[str], model : str, system_prompt : str = None):
    """Run parallel LLM call with a reference model."""
    responses = []
    for prompt in user_prompts:
        if 'gpt' in model:
            gpt_res = await prompt_model(prompt, model)
            responses.append(gpt_res)
        else:
            for sleep_time in [1, 2, 4]:
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
            
                    messages.append({"role": "user", "content": prompt})

                    response = await async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"}
                    )
                    responses.append(response.choices[0].message.content)
                    break
                except together.error.RateLimitError as e:
                    print(e)
                    await asyncio.sleep(sleep_time)
    return responses

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

def generate_fewshot_examples(n_shots, graph, mislead_dist, n, possible_people, objects):
    random.seed(47)
    examples = []
    objects = objects.copy()
    possible_people = possible_people.copy()
    random.shuffle(objects)
    random.shuffle(possible_people)
    pto = {p:o for p, o in zip(possible_people, objects)}

    for _ in range(n_shots):
        event_dict, max_actor = mislead_experiment(mislead_dist, n)
        storyboard = Storyboard('enters', graph, possible_people[:num_people], n, event_dict)

        sim = StorySimulator(
            people=possible_people[:num_people],
            locations=locations,
            action="enters",
            storyboard=storyboard,
            graph=graph
        )

        res = sim.run_simulation(n)
        story = sim.formal_to_story(res)
        label = storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]]

        # Form the example
        p0 = storyboard.actor_mapping["0"]
        p1 = storyboard.actor_mapping[max_actor]
        # First order mislead
        question_tom = f'Where does {p0} think {p1} is?'
        question_wm_human = f'When {p0} and {p1} were in the same location, where did {p0} watch {p1} go?'
        question_wm_inanimate = f'When {pto[p0].lower()} and {pto[p1].lower()} were last in the same room, where was {pto[p1].lower()} moved to?'
        inanimate_story = " ".join([pto[word] if word in pto else word for word in story.split(' ')])
        examples.append((f'{story}\nQ:{question_tom}\nA:{label}', f'{story}\nQ:{question_wm_human}\nA:{label}', f'{inanimate_story}\nQ:{question_wm_inanimate}\nA:{label}'))
    return examples
    

'''
Actual Experiements
'''
# Num of people
#values = [3,10,20,40]

# Mislead
values = [30]
#values = [30,40]
#values = values[-2:]

#models = ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
models = ["gpt-5.4"]

model_knowns = {m:{"T":[], "H":[], "I":[]} for m in models}
model_unkowns = {m:{"T":[], "H":[], "I":[]} for m in models}
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

objects = [
    "The shoe", "The ball", "The hat", 
    "The apple", "The chess piece", "The bottle",
    "The flashlight", "The cup", "The book",
    "The key", "The spoon", "The mirror",
    "The pillow", "The clock", "The camera",
    "The pen", "The notebook", "The chair",
    "The table", "The window", "The door",
    "The ladder", "The candle", "The lamp",
    "The basket", "The coin", "The brush",
    "The plate", "The bowl", "The bell",
    "The kite", "The statue", "The shell",
    "The map", "The blanket", "The drum",
    "The hammer", "The bucket", "The scarf",
    "The suitcase", "The envelope",
    "The sponge", "The comb", "The ring", "The whistle"
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
    num_trials = 100
    mislead_distance = values[v]

    random.seed(25)

    for _ in range(num_trials):
        
        ppl = random.sample(possible_people, num_people)
        event_dict, max_actor = second_order_tom_experiment(mislead_distance, story_length)
        storyboard = Storyboard('enters', graph, ppl, story_length, event_dict)

        sim = StorySimulator(
            people=ppl,
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
    intial_prompt = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother. There is enough information to answer every question."
    model_responses = {m:[] for m in models}    
    # Make a copy of each dataframe for each model
    dataframes = {model_name: df.copy() for model_name in models}
    
    tom_total, wm_total = 0, 0
    fewshot = False
    for i, d in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f'Processing {i}/{len(df)} using {models}')
        _ ,row = d
        p = row['P1'].split(',')
        # Create random mapping for objects
        global_pto = {p:o for p,o in zip(random.sample(possible_people, len(possible_people)),
                                          random.sample(objects, len(objects)))}
        # if fewshot:
        #     num_examples = 3
        #     examples = generate_fewshot_examples(num_examples, graph, values[v], story_length, possible_people, objects)

        #     tom_examples = '\n'.join([e[0] for e in examples])
        #     human_examples = '\n'.join([e[1] for e in examples])
        #     inanimate_examples = '\n'.join([e[2] for e in examples])

        #     # Setup prompts
        #     prompt_template = "{}\n{}\n{}\nQ: {}\nA:"
        #     tom_prompt = prompt_template.format(intial_prompt, tom_examples, row['Story'],f'Where does {p[0]} think {row["P2"]} is?')
        #     wm_human_prompt = prompt_template.format(intial_prompt, human_examples, row['Story'],f'When {p[0]} and {row["P2"]} were last in the same room, where does {row["P2"]} watch {p[0]} go?')
        #     mapped_p0, mapped_p2 = global_pto[p[0]].lower(), global_pto[row["P2"]].lower()
        #     wm_inanimate_prompt = prompt_template.format(intial_prompt, inanimate_examples,
        #                                                   " ".join(map(lambda x: global_pto.get(x, x), row['Story'].split(' '))),
        #                                                   f'When {mapped_p0} and {mapped_p2} were last in the same room, where was {mapped_p2} moved to?')
        # else:
        #     # Zero Shot
        prompt_template = "{}\n{}\nQ: {}"
        # tom_prompt = prompt_template.format(intial_prompt, row["Story"], f'Where does {p[0]} think {row["P2"]} is?')
        tom_prompt = prompt_template.format(intial_prompt, f'{row["Story"]}.', f'Where does {p[0]} think {p[1]} thinks {row["P2"]} is?')
        wm_human_prompt = prompt_template.format(intial_prompt, f'{row["Story"]}.', f'When {p[0]} and {p[1]} were last in the same room as {row["P2"]}, where did {row["P2"]} go?')
        mapped_p0, mapped_p2 = global_pto[p[0]], global_pto[row["P2"]]
        wm_inanimate_prompt = prompt_template.format(intial_prompt,
                                                        " ".join(map(lambda x: global_pto.get(x, x), f"{row['Story']}.".split(' '))),
                                                        f'When {mapped_p0.lower()} and {mapped_p2.lower()} were last in the same room, where was {mapped_p2.lower()} moved to?')
        #print(prompt)
        parallelized = False
        # if parallelized:
        #     async def _gather_tasks():
        #         coros = [run_llm_parallel(user_prompts=(tom_prompt, wm_human_prompt, wm_inanimate_prompt), model=model) for model in models]
        #         return await asyncio.gather(*coros)

        #     # Run it without a main() wrapper
        #     results = asyncio.run(_gather_tasks())  
        #     for model in models:
        #         model_responses[model].append(results)
        # else:
        for model in models:
            results = [prompt_model_synchronous(p, model) for p in (tom_prompt, wm_human_prompt, wm_inanimate_prompt)]
            model_responses[model].append(results)
    # Collect model responses for storage and accuracy
    out_dir = os.path.expanduser('~/scratch/mislead-results')
    os.makedirs(out_dir, exist_ok=True)
    for idx, model_name in enumerate(models):
        model_df = dataframes[model_name]
        model_df['TOM Responses'] = [model_res[0] for model_res in model_responses[model_name]]
        model_df['WM Human'] = [model_res[1] for model_res in model_responses[model_name]]
        model_df['WM Inanimate'] = [model_res[2] for model_res in model_responses[model_name]]
        # Check scores
        keys = ['TOM Responses', 'WM Human', 'WM Inanimate']
        letter = ["T", "H", "I"]
        for l, key in zip(letter, keys):
            outs= model_df.apply(lambda x: compute_score_unsure(x['Label'], x[key]), axis=1)
            known = [k for k in outs if k == 'True' or k == 'False']
            model_knowns[model_name][l] = (known)
            unknown = [k for k in outs if k != 'True' and k != 'False']
            model_unkowns[model_name][l] = (unknown)
            print(f"UKNOWN: {len(unknown)}")
        safe_name = model_name.replace('/', '_')
        model_df.to_csv(os.path.join(out_dir, f'{safe_name}_{values[v]}-mislead.csv'))
    

for mn in model_knowns.keys():
    for i in range(len(values)):
        for l in ["T", "H", "I"]:
            score = sum([1 for k in model_knowns[mn][l] if k == 'True'])
            print(f'Model {mn}|  Mislead = {values[i]}: {score}/{num_trials}, {len(model_unkowns[mn][l])} unkown for {l}')
