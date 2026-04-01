"""
Generate 50 examples using the mislead experiment setup with varying mislead distances.
Includes validation that both characters in the question appear in the story.
"""

from storyboard import Storyboard
from storysim import StorySimulator
from experiment_defs import mislead_experiment
import pandas as pd
import random
import os

# Setup
possible_people = [
    "Alice", "Bob", "Charlie", "Danny", "Edward",
    "Frank", "Georgia", "Hank", "Isaac", "Jake",
    "Kevin", "Liam", "Mia", "Nina", "Oliver",
    "Paula", "Quinn", "Rachel", "Steve", "Tina",
    "Uma", "Victor", "Wendy", "Xander", "Yara",
    "Zane", "Amber", "Brandon", "Carmen", "Derek",
    "Elena", "Felix", "Grace", "Harvey", "Ivy",
    "Jasmine", "Kyle", "Leah", "Miles", "Naomi",
    "Henry", "Wyatt", "Jose", "Neil", "Seth"
]

graph = { 
    "room_1": ["room_2", "the_hallway", "room_5"],
    "room_2": ["room_1", "room_3", "the_hallway"],
    "room_3": ["room_2", "room_4", "the_hallway"],
    "room_4": ["room_3", "room_5", "room_1"],
    "room_5": ["room_4", "room_1", "room_2"],
    "the_hallway": ["room_1", "room_4", "room_2"]
}

locations = list(graph.keys())
num_people = 7
story_length = 100
target_examples = 50

# Vary mislead distances
mislead_distances = [5,10, 20, 30, 40, 50, 60, 70, 80]

# Create dataframe to store examples
df = pd.DataFrame({
    'Story': [],
    'Label': [],
    'P1': [],
    'P2': [],
    'Character_1': [],
    'Character_2': [],
    'Mislead_Distance': [],
    'Question_Answerable': []
})

random.seed(25)
example_count = 0
distance_idx = 0

print(f'Generating {target_examples} examples with varying mislead distances...')

while example_count < target_examples:
    # Cycle through mislead distances
    mislead_distance = mislead_distances[distance_idx % len(mislead_distances)]
    distance_idx += 1
    
    try:
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
        
        # Get character information
        p1_actors = [storyboard.actor_mapping[str(a)] for a in range(int(max_actor))]
        p2_actor = storyboard.actor_mapping[max_actor]
        label_location = storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]]
        
        # Validation: Check both characters appear in the story
        p1_str = ','.join(p1_actors)
        
        # Convert story to lowercase for checking
        story_lower = story.lower()
        p1_in_story = all(name.lower() in story_lower for name in p1_actors)
        p2_in_story = p2_actor.lower() in story_lower
        
        if not (p1_in_story and p2_in_story):
            # Skip this example if characters aren't both in the story
            raise ValueError
        
        # Add valid example to dataframe
        new_row = pd.DataFrame({
            'Story': [story],
            'Label': [label_location],
            'P1': [p1_str],
            'P2': [p2_actor],
            'Character_1': [p1_str],
            'Character_2': [p2_actor],
            'Mislead_Distance': [mislead_distance],
            'Question_Answerable': [True]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        example_count += 1
        
        if example_count % 10 == 0:
            print(f'Generated {example_count}/{target_examples} examples')
        
    except Exception as e:
        print(f'Error generating example: {e}')
        continue

print(f'\nSuccessfully generated {len(df)} examples')
print(f'Mislead distance distribution:')
print(df['Mislead_Distance'].value_counts().sort_index())

# Save to CSV
out_dir = os.path.expanduser('~/scratch/mislead-examples-new')
os.makedirs(out_dir, exist_ok=True)

output_path = os.path.join(out_dir, 'sample.csv')
df.to_csv(output_path, index=False)
print(f'\nExamples saved to: {output_path}')

# Print sample
print(f'\nSample example:')
if len(df) > 0:
    sample = df.iloc[0]
    print(f"Mislead Distance: {sample['Mislead_Distance']}")
    print(f"P1: {sample['P1']}")
    print(f"P2: {sample['P2']}")
    print(f"Label (correct answer): {sample['Label']}")
    print(f"Story excerpt: {sample['Story'][:200]}...")
