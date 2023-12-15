import torch
import torch.distributions as dist
import torch.nn.functional as F


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given
    distribution, 0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


def generate_samples(model, start_sequence, i2w, w2i, max_length=50, samples=10):
    model.eval()
    with torch.no_grad():
        for _ in range(samples):
            input_seq = start_sequence.copy()
            for _ in range(max_length):
                input_tensor = torch.tensor([input_seq], dtype=torch.long)
                output = model(input_tensor)
                next_token_logits = output[0, -1, :]
                next_token = sample(next_token_logits)
                next_token_item = next_token.item()
                input_seq.append(next_token_item)
                if next_token_item == w2i[".end"]:
                    break
            print("".join([i2w[i] for i in input_seq]))
