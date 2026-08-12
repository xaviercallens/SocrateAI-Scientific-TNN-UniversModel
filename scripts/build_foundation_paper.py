import os
import unicodedata

def clean_for_latex(text):
    # Remove accents and normalize to ASCII safely
    normalized = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return normalized

def build_foundation_paper():
    original_tex_paths = {
        "en": "docs/research_paper/tnn_univers_model_paper_en.tex",
        "fr": "docs/research_paper/tnn_univers_model_paper_fr.tex"
    }
    new_tex_paths = {
        "en": "docs/research_paper/tnn_univers_model_foundation_paper_en.tex",
        "fr": "docs/research_paper/tnn_univers_model_foundation_paper_fr.tex"
    }
    
    markdown_files = [
        "specs/📚 Boîte à Outils pour l'Implémentation.md",
        "specs/Bases_de_Donnees_IA_Nature_Univers.md",
        "specs/Best_Practices_and_Hardness.md",
        "specs/Foundations of Poly-Algebraic Calculus.md",
        "specs/Manifesto TNN.md",
        "specs/Physics_Use_Cases.md",
        "specs/Repository_Leverage_Analysis.md",
        "specs/Roadmap_Implementation_Etape_par_Etape.md",
        "specs/Scientific_Audit_Ledger.md",
        "specs/Scientific_Rigor_Audit.md"
    ]
    
    appendix_content = ""
    for md_file in markdown_files:
        if not os.path.exists(md_file):
            print(f"File not found: {md_file}")
            continue
            
        with open(md_file, "r") as f:
            md_content = f.read()
            
        clean_title = os.path.basename(md_file).replace(".md", "").replace("_", " ")
        clean_title = clean_title.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
        clean_title = clean_for_latex(clean_title)
        
        appendix_content += f"\\subsection{{{clean_title}}}\n"
        appendix_content += "\\begin{lstlisting}[language={}, caption={}, basicstyle=\\ttfamily\\footnotesize, breaklines=true]\n"
        
        md_content = clean_for_latex(md_content)
            
        appendix_content += md_content
        appendix_content += "\n\\end{lstlisting}\n\n"
        
    for lang in ["en", "fr"]:
        with open(original_tex_paths[lang], "r") as f:
            content = f.read()
            
        parts = content.split("\\end{document}")
        main_body = parts[0]
        
        if lang == "fr":
            appendix_tex = "\n\\newpage\n\\appendix\n\\section{Documents de Fondation et Sp\\'{e}cifications}\nLes sections suivantes contiennent les specifications brutes originales en markdown, les audits et les plans architecturaux du TNN Univers Model.\n\n"
        else:
            appendix_tex = "\n\\newpage\n\\appendix\n\\section{Foundation Documents and Specifications}\nThe following sections contain the original raw markdown specifications, audits, and architectural blueprints for the TNN Univers Model.\n\n"
        
        final_content = main_body + appendix_tex + appendix_content + "\\end{document}\n"
        
        with open(new_tex_paths[lang], "w") as f:
            f.write(final_content)
            
        print(f"Generated {new_tex_paths[lang]} successfully.")

if __name__ == "__main__":
    build_foundation_paper()
