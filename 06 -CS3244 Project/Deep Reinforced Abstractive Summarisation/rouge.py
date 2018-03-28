class ROUGE:
    def score_n(self, reference, generated, n):
        reference = reference.split()
        generated = generated.split()
        reference_n_grams = {}
        generated_n_grams = {}

        for start in range(0, len(reference) - n + 1):
            n_gram = " ".join(reference[start : start + n])
            if n_gram not in reference_n_grams:
                reference_n_grams[n_gram] = 0
            reference_n_grams[n_gram] += 1

        for start in range(0, len(generated) - n + 1):
            n_gram = " ".join(generated[start : start + n])
            if n_gram not in generated_n_grams:
                generated_n_grams[n_gram] = 0
            generated_n_grams[n_gram] += 1

        # Calculate recall
        recall_hit = 0.0
        for key in reference_n_grams.keys():
            if key in generated_n_grams:
                recall_hit += reference_n_grams[key]
        recall_score = 0.0
        if len(reference) != 0:
            recall_score = recall_hit / len(reference)

        # Calculate precision
        precision_hit = 0.0
        for key in generated_n_grams.keys():
            if key in reference_n_grams:
                precision_hit += generated_n_grams[key]
        precision_score = 0.0
        if len(generated) != 0:
            precision_score = precision_hit / len(generated)
        
        return {"recall"   : recall_score,
                "precision": precision_score}
    
    def score(self, reference, generated):
        return {"rouge-1": self.score_n(reference, generated, 1),
                "rouge-2": self.score_n(reference, generated, 2)}

