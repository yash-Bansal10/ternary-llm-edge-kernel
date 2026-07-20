import os

def compile_paper():
    sections = [
        "Section_0_Abstract.md",
        "Section_1_Introduction.md",
        "Section_2_Material_and_Methods.md",
        "Section_3_Result_and_Discussion.md",
        "Section_4_Conclusion_and_References.md"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manuscript_dir = os.path.join(script_dir, "..", "manuscript")
    output_file = os.path.join(manuscript_dir, "Research_Paper_draft.md")
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Add a title at the very top
        outfile.write("# Bypassing the Unpacking Tax: A Bit-Serial Architecture for Ternary LLM Edge Inference\n\n")
        
        for sec in sections:
            sec_path = os.path.join(manuscript_dir, sec)
            if os.path.exists(sec_path):
                with open(sec_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n\n")
                print(f"✅ Added {sec}")
            else:
                print(f"❌ Warning: {sec} not found.")

    print(f"\n🎉 Successfully compiled all sections into: Research_Paper_draft.md")

if __name__ == "__main__":
    compile_paper()
