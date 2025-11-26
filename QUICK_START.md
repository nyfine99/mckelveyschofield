# Quick Start Guide

Get this project running in just a few minutes, and see the electoral dynamics in action!

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/nyfine99/mckelveyschofield.git
cd mckelveyschofield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a demo (creates output folder automatically)
python -m scripts.euclidean_electorate
```

Note: to create any animations, you will need to install ffmpeg on your system. The link to do so can be found [here](https://ffmpeg.org/download.html).

## What You'll See

Running the demo will generate:
- **Winset boundary** - policies that can defeat the incumbent
- **McKelvey-Schofield path** - strategic policy transition sequence
- **Animation** showing the path through policy space

These can all be found in the newly-created output folder.

## Explore Different Scenarios

### McKelvey-Schofield in a Polarized Electorate
```bash
python -m scripts.polarized_electorate
```

### Run Many Simulations
```bash
python -m scripts.multirun_euclidean
```

### Ranked Choice Voting
```bash
python -m scripts.small_rcv_electorate
```

### New Policy Entry Optimization
```bash
python -m scripts.us_electorate_scripts.echelon_electorate_rcv
```

## Understanding the Outputs

### Winset Boundary Plot
- **Black dots**: Individual voters
- **Orange Circle**: Incumbent policies
- **Shaded region**: The set of policies that can defeat the incumbent

### McKelvey-Schofield Path Plot
- **Black dots**: Individual voters
- **Blue X**: Initial policy
- **Red star**: Goal policy
- **Green dots/arrows**: Strategic policy transitions for agenda setters to manipulate outcomes

### McKelvey-Schofield Path Animation
- **Small dots**: Individual voters, colored by choice
- **Large dots**: Current policy options for voters
- **Blue X**: Initial policy
- **Red star**: Goal policy

### Ranked-Choice Voting Outputs
- **Round-by-round animation**: Animates elimination process
- **Sankey diagram**: Visualizes vote flow between candidates

## Customization

### Modify Voter Distributions
Edit `scripts/euclidean_electorate.py`:
```python
# Change voter distribution parameters
for i in range(100):
    voters.append(SimpleVoter(Policy(np.array([
        gauss(50, 15),  # Mean 50, std 15 for issue 1
        gauss(50, 10)   # Mean 50, std 10 for issue 2
    ]))))
```

### Adjust Policy Positions
```python
# Modify policy coordinates
p1 = Policy([45, 50], "Centrism")    # [issue1, issue2]
p2 = Policy([80, 90], "Extremism")
```

### Change Simulation Parameters
```python
# Adjust winset computation precision
electorate.plot_winset_boundary(
    p1,
    n_directions=360,           # More directions = smoother boundary
    n_halving_iterations=12,    # More iterations = more precise boundary
    output_folder="output",
    filename="custom_winset.png"
)
```

## Troubleshooting

### Common Issues

**"FFmpeg not found" (for animations)**
- **Windows**: Download from https://ffmpeg.org/download.html
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### Performance Tips

- **Small electorates** (< 1000 voters) run quickly
- **Large simulations** (> 10000 voters) may take several minutes
- **Animations** require more memory and processing time

## Next Steps

1. **Read the docs**: Check `docs/mckelvey_schofield.md` and `docs/ranked_choice_voting.md` for a more detailed dive into the capabilities offered in each regard
2. **Explore scripts**: Try different parameter combinations
3. **Modify code**: Add new voter types or electoral systems

---

*Questions? Check the main README or open an issue on GitHub!* 