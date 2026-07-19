import torch
import torch.nn.functional as F
import json
import faiss
import numpy as np
from pathlib import Path
from .model import SASRec


class Recommender:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")
        
        with open(self.model_dir / "hyperparams.json", "r") as f:
            self.hyperparams = json.load(f)
            
        with open(self.model_dir / "vocab.json", "r") as f:
            vocab_data = json.load(f)
            self.item2idx = vocab_data["item2idx"]
            self.idx2item = {int(k): v for k, v in vocab_data["idx2item"].items()}
            
        self.model = SASRec(
            vocab_size=self.hyperparams["vocab_size"],
            embed_dim=self.hyperparams["embed_dim"],
            num_heads=self.hyperparams["num_heads"],
            num_layers=self.hyperparams["num_layers"],
            max_seq_len=self.hyperparams["max_seq_len"],
            dropout=self.hyperparams["dropout"]
        )
        
        weights_path = self.model_dir / "model_weights.pth"
        state_dict = torch.load(weights_path, map_location=self.device)
        
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("transformer_encoder.layers."):
                new_key = key.replace("transformer_encoder.layers.", "encoder_layers.")
                remapped_state_dict[new_key] = value
            else:
                remapped_state_dict[key] = value
                
        self.model.load_state_dict(remapped_state_dict)
        self.model.eval()
        
        index_path = self.model_dir / "faiss_index.index"
        self.faiss_index = faiss.read_index(str(index_path))
        
        self.question_clusters = self._load_clusters()
        self.item_to_cluster = {}
        for cluster_id, items in self.question_clusters.items():
            for item in items:
                self.item_to_cluster[item] = int(cluster_id)
        
        print(f"✅ Recommender initialized. Model: {self.hyperparams['best_model']}")
        print(f"   Loaded {len(self.question_clusters)} topics from cluster_info.json")

    def _load_clusters(self):
        cluster_info_path = self.model_dir / "cluster_info.json"
        
        if cluster_info_path.exists():
            with open(cluster_info_path, "r") as f:
                cluster_info = json.load(f)
            print(f"   📊 Optimal k: {cluster_info['optimal_k']}")
            print(f"    Silhouette Score: {cluster_info['silhouette_score']:.3f}")
            return cluster_info["clusters"]
        else:
            print("   ️ cluster_info.json not found, using fallback k=10")
            return {str(i): [] for i in range(10)}

    def get_recommendations(self, item_names: list, top_k: int = 5):
        item_indices = []
        for name in item_names:
            if name in self.item2idx:
                item_indices.append(self.item2idx[name])
                
        if not item_indices:
            return []
            
        max_len = self.hyperparams["max_seq_len"]
        if len(item_indices) > max_len:
            item_indices = item_indices[-max_len:]
            
        input_seq = torch.tensor([item_indices], dtype=torch.long).to(self.device)
        mask = torch.ones_like(input_seq, dtype=torch.bool)
        
        with torch.no_grad():
            session_emb = self.model.get_session_embedding(input_seq, mask)
            
        session_emb_np = session_emb.cpu().numpy()
        distances, indices = self.faiss_index.search(session_emb_np, top_k + 10)
        
        recommendations = []
        seen = set(item_indices)
        
        for idx in indices[0]:
            vocab_idx = idx + 2
            
            if vocab_idx in seen:
                continue
            if vocab_idx not in self.idx2item:
                continue
                
            item_name = self.idx2item[vocab_idx]
            cluster_id = self.item_to_cluster.get(item_name, -1)
            
            recommendations.append({
                "item_name": item_name,
                "cluster_id": cluster_id,
                "score": float(distances[0][list(indices[0]).index(idx)])
            })
            
            if len(recommendations) >= top_k:
                break
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

    def get_explanation(self, item_names: list, recommended_item: str, top_k_influence: int = 2):
        item_indices = []
        for name in item_names:
            if name in self.item2idx:
                item_indices.append(self.item2idx[name])
        
        if not item_indices:
            return None
        
        max_len = self.hyperparams["max_seq_len"]
        if len(item_indices) > max_len:
            item_indices = item_indices[-max_len:]
        
        input_seq = torch.tensor([item_indices], dtype=torch.long).to(self.device)
        mask = torch.ones_like(input_seq, dtype=torch.bool)
        
        with torch.no_grad():
            session_emb = self.model.get_session_embedding(input_seq, mask)
            
            rec_idx = self.item2idx.get(recommended_item)
            if rec_idx is None:
                return None
            
            rec_emb = self.model.item_embedding(torch.tensor([[rec_idx]], dtype=torch.long).to(self.device))
            rec_emb = F.normalize(rec_emb, p=2, dim=-1)
            
            history_embs = self.model.item_embedding(input_seq)
            history_embs = F.normalize(history_embs, p=2, dim=-1)
            
            similarity = torch.bmm(rec_emb, history_embs.transpose(1, 2))
            similarity = similarity.squeeze(0).squeeze(0)
            
            attention_weights, _ = self.model.get_attention_weights(input_seq, mask)
            last_layer_attn = attention_weights[-1, 0]
            
            if last_layer_attn.dim() == 3:
                avg_attn = last_layer_attn.mean(dim=0)
            else:
                avg_attn = last_layer_attn
            
            influence_weights = avg_attn[-1, :-1]
            influence_weights = influence_weights / influence_weights.sum()
        
        top_sim_indices = similarity.topk(min(top_k_influence, len(similarity))).indices
        
        influential_questions = []
        for idx in top_sim_indices:
            question_name = item_names[idx]
            sim_score = similarity[idx].item()
            attn_weight = influence_weights[idx].item()
            influential_questions.append({
                'question': question_name,
                'similarity': sim_score,
                'attention_weight': attn_weight
            })
        
        rec_cluster = self.item_to_cluster.get(recommended_item, -1)
        
        if influential_questions:
            top_question = influential_questions[0]['question']
            top_sim = influential_questions[0]['similarity']
            
            explanation = f"Вопрос `{recommended_item}` рекомендован, потому что он наиболее похож на вопрос `{top_question}` из вашей истории (сходство: {top_sim*100:.1f}%)."
            
            if len(influential_questions) > 1:
                second_question = influential_questions[1]['question']
                second_sim = influential_questions[1]['similarity']
                explanation += f" Также высокое сходство с вопросом `{second_question}` ({second_sim*100:.1f}%)."
            
            top_cluster = self.item_to_cluster.get(top_question, -1)
            if top_cluster == rec_cluster:
                explanation += f" Оба вопроса относятся к одной теме (Тема {rec_cluster}), что указывает на логическое продолжение изучения материала."
            else:
                explanation += f" Вопросы относятся к разным темам (рекомендуемый: Тема {rec_cluster}, похожий: Тема {top_cluster}), что может указывать на междисциплинарную связь."
        else:
            explanation = "Рекомендация основана на общем паттерне вашей траектории обучения."
        
        return {
            'explanation': explanation,
            'influential_questions': influential_questions,
            'recommended_cluster': rec_cluster
        }

    def get_all_items(self):
        special_tokens = {'[PAD]', '[MASK]'}
        return [item for item in self.item2idx.keys() if item not in special_tokens]

    def get_items_by_cluster(self):
        return self.question_clusters