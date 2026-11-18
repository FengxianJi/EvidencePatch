from method.common import call_model, parse_json_object, result

SYSTEM = "Verify review claims only against the provided manuscript. Treat retrieval as evidence, not proof."


def run(context):
    claims = context["claims"] if "claims" in context else context.get("reviews", "")
    if not claims:
        return result("peerify_verify", "", verifications=[], calls=0,
                      interpretation="no eligible claim was present in the frozen candidate pool")
    prompt = f"""For each review claim, return a status supported|partly_supported|not_supported|not_determined and exact manuscript quotes. Do not draft a patch.
MANUSCRIPT:\n{context.get('manuscript', '')}\nCLAIMS/REVIEWS:\n{claims}\nReturn JSON {{\"verifications\":[{{\"claim\":\"...\",\"status\":\"...\",\"evidence_quotes\":[\"...\"],\"confidence\":0.0}}]}}."""
    obj = parse_json_object(call_model(context, SYSTEM, prompt, "baseline_claim_verification"))
    return result("peerify_verify", "", verifications=obj.get("verifications", []), calls=1,
                  provenance="Peerify-style claim decomposition/retrieval/verification adaptation")
