import os
import unicodedata

def clean_for_latex(text):
    # Remove accents and normalize to ASCII safely
    normalized = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return normalized

def build_foundation_paper():
    original_tex_path = "docs/research_paper/tnn_univers_model_paper.tex"
    new_tex_path = "docs/research_paper/tnn_univers_model_foundation_paper.tex"
    
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
    
    with open(original_tex_path, "r") as f:
        content = f.read()
        
    # Split before \end{document}
    parts = content.split("\\end{document}")
    if len(parts) != 2:
        print("Error parsing tex file.")
        return
        
    main_body = parts[0]
    
    appendix_tex = "\n\\newpage\n\\appendix\n\\section{Foundation Documents and Specifications}\n"
    appendix_tex += "The following sections contain the original raw markdown specifications, audits, and architectural blueprints for the TNN Univers Model.\n\n"
    
    for md_file in markdown_files:
        if not os.path.exists(md_file):
            print(f"File not found: {md_file}")
            continue
            
        with open(md_file, "r") as f:
            md_content = f.read()
            
        # Clean title for LaTeX section
        clean_title = os.path.basename(md_file).replace(".md", "").replace("_", " ")
        # Escape some chars just for the title
        clean_title = clean_title.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
        clean_title = clean_for_latex(clean_title)
        
        appendix_tex += f"\\subsection{{{clean_title}}}\n"
        appendix_tex += "\\begin{lstlisting}[language={}, caption={}, basicstyle=\\ttfamily\\footnotesize, breaklines=true]\n"
        
        md_content = clean_for_latex(md_content)
            
        # In lstlisting, special characters are safe.
        appendix_tex += md_content
        appendix_tex += "\n\\end{lstlisting}\n\n"
        
    final_content = main_body + appendix_tex + "\\end{document}\n"
    
    with open(new_tex_path, "w") as f:
        f.write(final_content)
        
    print(f"Generated {new_tex_path} successfully.")

if __name__ == "__main__":
    build_foundation_paper()
