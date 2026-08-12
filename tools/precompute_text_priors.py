import argparse
from pathlib import Path
import torch
import torch.nn.functional as F


def _read_non_empty_lines(path: Path):
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def parse_actions_file(actions_path: Path):
    """
    Parse verb/action file lines like:
    ride
    hold
    ...
    Return ordered verb list.
    """
    return [x.strip().lower() for x in _read_non_empty_lines(actions_path)]


def parse_interaction_prompt_file(prompt_path: Path):
    """
    Parse prompt lines like:
    001  airplane        board
    Return list of dicts: [{'id': 1, 'object': 'airplane', 'verb': 'board'}, ...]
    """
    rows = []
    for raw in _read_non_empty_lines(prompt_path):
        parts = raw.split()
        if len(parts) < 3:
            continue
        idx = int(parts[0])
        obj = parts[1].strip().lower()
        verb = parts[2].strip().lower()
        rows.append({'id': idx, 'object': obj, 'verb': verb})
    return rows


def parse_interaction_sentence_file(sentence_path: Path):
    """
    Parse sentence lines like:
    001 A person boards an airplane.
    Return dict: {1: 'A person boards an airplane.', ...}
    """
    id_to_sent = {}
    for raw in _read_non_empty_lines(sentence_path):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        idx = int(parts[0])
        sent = parts[1].strip()
        id_to_sent[idx] = sent
    return id_to_sent


def merge_interaction_prompt_and_sentence(prompt_rows, id_to_sent):
    merged = []
    for row in prompt_rows:
        idx = row['id']
        if idx not in id_to_sent:
            raise ValueError(f'Missing sentence for interaction id={idx}')
        merged.append({
            'id': idx,
            'object': row['object'],
            'verb': row['verb'],
            'interaction': id_to_sent[idx],
        })
    return merged


def write_merged_interaction_file(merged_rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"object:{r['object']} verb:{r['verb']} interaction:{r['interaction']}"
        for r in merged_rows
    ]
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


@torch.no_grad()
def encode_text_with_real_blip2(texts, model_name, device, batch_size=16):
    """
    BLIP-2 text encoding by frozen language token embeddings + masked mean pooling.
    """
    from transformers import Blip2Processor, Blip2Model

    processor = Blip2Processor.from_pretrained(model_name, local_files_only=True)
    model = Blip2Model.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_feats = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[BLIP2-Text] device={device}, batch_size={batch_size}, total_texts={len(texts)}, total_batches={total_batches}")

    for batch_idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[i:i + batch_size]

        proc = processor(text=batch_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = proc['input_ids'].to(device)
        attention_mask = proc['attention_mask'].to(device)

        tok = model.language_model.get_input_embeddings()(input_ids)  # [B,T,D]
        att = attention_mask.unsqueeze(-1).float()
        feat = (tok * att).sum(dim=1) / att.sum(dim=1).clamp(min=1e-6)

        all_feats.append(feat.detach().cpu())
        print(f"[BLIP2-Text] processed batch {batch_idx}/{total_batches} ({len(batch_texts)} texts)")

    text_features = torch.cat(all_feats, dim=0)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


def build_verb_only_priors(verbs, model_name, device, output_dim=256):
    """
    Verb-only prior building:
    - verb_text_features: verb embeddings [N_verb, output_dim]
    - verb_sim_matrix: verb-verb similarity [N_verb, N_verb]
    """
    verb_features = encode_text_with_real_blip2(verbs, model_name=model_name, device=device)
    if verb_features.shape[-1] != output_dim:
        verb_features = verb_features[:, :output_dim] if verb_features.shape[-1] > output_dim else F.pad(verb_features, (0, output_dim - verb_features.shape[-1]))
    verb_features = F.normalize(verb_features, dim=-1)
    verb_sim = verb_features @ verb_features.t()
    verb_to_idx = {v: i for i, v in enumerate(verbs)}

    return {
        'verbs': verbs,
        'verb_text_features': verb_features,
        'verb_sim_matrix': verb_sim,
        'verb_to_idx': verb_to_idx,
    }


def build_sentence_only_priors(merged_rows, model_name, device, output_dim=256):
    """
    Sentence-only prior building:
    - text_features: interaction sentence embeddings [N_interaction, output_dim]
    - sim_matrix: interaction-interaction similarity [N_interaction, N_interaction]

    Also exports interaction index lookup for Top-1/2/3 object-verb pairs:
    - interaction_index_map: dict[str, int] with key "object#verb"
    """
    interactions = [r['interaction'] for r in merged_rows]
    objects = [r['object'] for r in merged_rows]
    verbs = [r['verb'] for r in merged_rows]

    text_features = encode_text_with_real_blip2(interactions, model_name=model_name, device=device)
    if text_features.shape[-1] != output_dim:
        text_features = text_features[:, :output_dim] if text_features.shape[-1] > output_dim else F.pad(text_features, (0, output_dim - text_features.shape[-1]))
    text_features = F.normalize(text_features, dim=-1)
    sim_matrix = text_features @ text_features.t()

    interaction_index_map = {
        f"{obj}#{verb}": idx
        for idx, (obj, verb) in enumerate(zip(objects, verbs))
    }

    unique_objects = sorted(set(objects))
    unique_verbs = sorted(set(verbs))
    obj_to_idx = {o: i for i, o in enumerate(unique_objects)}
    verb_to_idx = {v: i for i, v in enumerate(unique_verbs)}

    # map each (object_idx, verb_idx) to interaction sentence index
    obj_verb_to_interaction = torch.full((len(unique_objects), len(unique_verbs)), -1, dtype=torch.long)
    for idx, (obj, verb) in enumerate(zip(objects, verbs)):
        obj_verb_to_interaction[obj_to_idx[obj], verb_to_idx[verb]] = idx

    return {
        'source': 'interaction_sentence_only',
        'interactions': interactions,                # interaction sentence list (class order)
        'objects': objects,                          # aligned with interactions
        'verbs': verbs,                              # aligned with interactions
        'text_features': text_features,              # [N_interaction, D]
        'sim_matrix': sim_matrix,                    # interaction x interaction
        'interaction_index_map': interaction_index_map,
        'unique_objects': unique_objects,
        'unique_verbs': unique_verbs,
        'obj_to_idx': obj_to_idx,
        'verb_to_idx': verb_to_idx,
        'obj_verb_to_interaction': obj_verb_to_interaction,
        # Backward-compatible alias
        'interaction_sim_matrix': sim_matrix,
    }


def main():
    parser = argparse.ArgumentParser('Precompute sentence-only textual priors for interaction classes')
    parser.add_argument('--interaction_prompt_file', required=True, type=str,
                        help='Path to interaction prompt list (object+verb), e.g. tools/interaction-prompt.txt')
    parser.add_argument('--interaction_sentence_file', required=True, type=str,
                        help='Path to interaction sentence list, e.g. tools/interaction-sentence.txt')
    parser.add_argument('--actions_file', default='', type=str,
                        help='Optional path to actions/verbs list (one verb per line), e.g. tools/actions.txt')
    parser.add_argument('--verb_output_path', default='', type=str,
                        help='Optional output .pt path for verb-only priors (separate file).')
    parser.add_argument('--merged_output_file', default='', type=str,
                        help='Optional path to save merged text lines in required format')
    parser.add_argument('--output_path', required=True, type=str,
                        help='Output .pt path for sentence-only priors')
    parser.add_argument('--blip2_model_name', default='BLIP2-opt-2.7b', type=str,
                        help='Local BLIP-2 model folder path or HF model id')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    if args.actions_file and not args.verb_output_path:
        raise ValueError('When --actions_file is provided, --verb_output_path must also be provided for separate verb-only output.')

    prompt_rows = parse_interaction_prompt_file(Path(args.interaction_prompt_file))
    id_to_sent = parse_interaction_sentence_file(Path(args.interaction_sentence_file))
    merged_rows = merge_interaction_prompt_and_sentence(prompt_rows, id_to_sent)

    if args.merged_output_file:
        write_merged_interaction_file(merged_rows, Path(args.merged_output_file))
        print(f"Saved merged interaction text file to: {args.merged_output_file}")

    output = build_sentence_only_priors(
        merged_rows=merged_rows,
        model_name=args.blip2_model_name,
        device=args.device,
        output_dim=256,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    print(f'Saved interaction textual priors to: {out_path}')
    print(f"Num interactions: {len(output['interactions'])}, feature dim: {output['text_features'].shape[-1]}")
    print(f"sim_matrix shape: {tuple(output['sim_matrix'].shape)}")
    print(f"interaction_index_map size: {len(output['interaction_index_map'])}")

    # Optional: compute verb-only priors/similarity and save to a separate file.
    if args.actions_file and args.verb_output_path:
        verbs = parse_actions_file(Path(args.actions_file))
        if len(verbs) > 0:
            verb_output = build_verb_only_priors(
                verbs=verbs,
                model_name=args.blip2_model_name,
                device=args.device,
                output_dim=256,
            )
            verb_out_path = Path(args.verb_output_path)
            verb_out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(verb_output, verb_out_path)
            print(f"Saved verb-only priors to: {verb_out_path}")
            print(f"Num verbs: {len(verb_output['verbs'])}, verb_sim_matrix shape: {tuple(verb_output['verb_sim_matrix'].shape)}")


if __name__ == '__main__':
    main()
