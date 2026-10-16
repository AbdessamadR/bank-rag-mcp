
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.retrieval.retriever import BankReportRetriever


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"


class BankReportAnswerGenerator:
    """Génère des réponses à partir des rapports bancaires récupérés."""

    def __init__(
        self,
        retriever=None,
        model_name: str = MODEL_NAME,
        ollama_url: str = OLLAMA_URL,
    ):
        self.retriever = retriever or BankReportRetriever()
        self.model_name = model_name
        self.ollama_url = ollama_url

    def _call_llm(self, question: str, passages: list[dict]) -> str:
        context_parts = []

        for index, passage in enumerate(passages, start=1):
            context_parts.append(
                f"[Extrait {index}]\n"
                f"Source : {passage.get('source') or 'Inconnue'}\n"
                f"Année : {passage.get('year') or 'Inconnue'}\n"
                f"Page : {passage.get('page') or 'Inconnue'}\n"
                f"Contenu : {passage['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
Tu es un assistant spécialisé dans l'analyse financière bancaire.

Règles impératives :
1. Réponds en français.
2. Utilise uniquement les extraits fournis.
3. N'invente aucun chiffre, fait ou référence.
4. Si la réponse n'est pas présente dans les extraits,
   indique clairement que les documents fournis ne permettent
   pas de répondre avec certitude.
5. Distingue les résultats consolidés des résultats sociaux.
6. Si la source indique explicitement un taux, reprends ce taux.
   Ne le remplace pas par un calcul effectué sur des chiffres arrondis.
7. Explique brièvement le calcul si la question le demande et
   si les données nécessaires sont disponibles.
8. Cite les pages utilisées sous la forme [Source, année, p. X].
9. Ne prétends pas avoir consulté une page qui n'est pas fournie.
"""

        user_prompt = (
            f"Question : {question}\n\n"
            f"Extraits des rapports :\n{context}\n\n"
            "Rédige une réponse précise et concise avec ses références."
        )

        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }

        request = Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))

            return result["message"]["content"].strip()

        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Erreur Ollama HTTP {exc.code}: {details}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Impossible de contacter Ollama. "
                "Vérifie que le serveur fonctionne sur le port 11434."
            ) from exc

        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "La réponse reçue d'Ollama est invalide."
            ) from exc

    def answer(
        self,
        question: str,
        top_k: int = 5,
        year: str | None = None,
    ) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("La question ne peut pas être vide.")

        passages = self.retriever.search(
            question=question,
            top_k=top_k,
            year=year,
        )

        if not passages:
            return {
                "question": question,
                "answer": (
                    "Je n'ai trouvé aucun passage pertinent "
                    "dans les rapports."
                ),
                "sources": [],
            }

        answer_text = self._call_llm(question, passages)

        # Les références sont construites à partir des métadonnées
        # réelles retournées par le retriever.
        sources = []
        seen = set()

        for passage in passages:
            source = passage.get("source")
            source_year = passage.get("year")
            page = passage.get("page")

            key = (source, source_year, page)

            if key not in seen and any(value is not None for value in key):
                seen.add(key)
                sources.append({
                    "source": source,
                    "year": source_year,
                    "page": page,
                })

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
        }


if __name__ == "__main__":
    generator = BankReportAnswerGenerator()

    question = (
        "Quel est le produit net bancaire consolidé "
        "en 2024 et quelle est sa progression par rapport à 2023 ?"
    )

    result = generator.answer(
        question=question,
        top_k=5,
        year="2024",
    )

    print("\nQUESTION")
    print(result["question"])

    print("\nRÉPONSE")
    print(result["answer"])

    print("\nSOURCES RETROUVÉES")
    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"Année : {source['year']} | "
            f"Page : {source['page']}"
        )
