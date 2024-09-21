import os
import re
import PyPDF2
from sync_embedding import update_embeddings
import argparse



def remove_tail(text):
    references = [m.start() for m in re.finditer(r'\breference(?:s)?\b', text, re.IGNORECASE)]

    if references:
        # Remove everything after the last occurrence of "reference(s)"
        return text[:references[-1]]
    else:
        # If "reference" or "references" is not found, return the original text
        return text


def consolidate_paragraphs(text):
    """
    Consolidate fragmented lines into paragraphs.

    :param text: The extracted text from the PDF.
    :return: Text with consolidated paragraphs.
    """

    # remove references
    text = remove_tail(text)

    # Replace multiple consecutive line breaks with a unique placeholder
    text = re.sub(r'\n\s*\n', '<PARAGRAPH_BREAK>', text)

    # Replace remaining single line breaks with spaces
    text = re.sub(r'\n', ' ', text)

    # Replace the unique placeholder with double line breaks to indicate paragraph breaks
    text = text.replace('<PARAGRAPH_BREAK>', '\n\n')

    return text


def convert_pdf_to_txt(pdf_path, output_folder):
    """
    Convert a PDF file to a text file with consolidated paragraphs.

    :param pdf_path: Path to the input PDF file.
    :param txt_path: Path to the output text file.
    """
    try:
        # Open the PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Initialize an empty string to store text
            text = ""

            # Iterate through all the pages
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        # Consolidate paragraphs
        consolidated_text = consolidate_paragraphs(text)

        # Write the consolidated text to the output file
        filename = os.path.basename(pdf_path)
        txt_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            text_file.write(consolidated_text)

        print(f"Successfully converted {filename}")

    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def remove_txt_file(output_txt_folder, all_pdf_folder, sub_pdf_folder):
    """
    Remove text files in the output folder if their corresponding PDF files
    are not present in the source folders.

    :param output_folder: Path to the folder containing text files.
    :param all_folder: Path to the main folder containing subfolders with PDF files.
    :param sub_folder: List of subfolder names containing PDF files.
    """

    delete_count = 0

    # Get all text files in the output folder
    text_files = [f for f in os.listdir(output_txt_folder) if f.lower().endswith('.txt')]

    # Create a set of all PDF files in the source folders
    pdf_files = set()
    for folder in sub_pdf_folder:
        pdf_folder = os.path.join(all_pdf_folder, folder)
        pdf_files.update(f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf'))

    for text_file in text_files:
        pdf_file = text_file.replace('.txt', '.pdf')

        # Check if the corresponding PDF exists in the set
        if pdf_file not in pdf_files:
            text_path = os.path.join(output_txt_folder, text_file)
            try:
                os.remove(text_path)
                delete_count += 1
                print(f"Removed {text_file} as corresponding PDF was not found.")
            except Exception as e:
                print(f"Error removing {text_file}: {e}")

    return delete_count


def main(args):
    all_folder = r"C:\Users\lanzh\OneDrive - Nanyang Technological University\PhD Papers\all_papers"
    sub_folder = ["AI_technique", "AI_application", "IS_tool", "IS_study", "Sustainability general"]
    output_folder = "papers_text"
    new_files = []

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    add_count = 0

    # Iterate through all PDF files in the source folder and convert to txt file
    print("Checking files to add...")
    for folder in sub_folder:
        pdf_folder = os.path.join(all_folder, folder)
        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                # check if it is a new file
                txt_path = os.path.join(output_folder, file.replace(".pdf", ".txt"))
                if not os.path.exists(txt_path):
                    pdf_path = os.path.join(pdf_folder, file)
                    convert_pdf_to_txt(pdf_path, output_folder)
                    new_files.append(txt_path)
                    add_count += 1


    # for txt file that does not have corresponding pdf file, remove the file
    print("Checking files to delete...")
    delete_count = remove_txt_file(output_folder, all_folder, sub_folder)

    # report
    total_files = len([f for f in os.listdir(output_folder) if f.endswith('.txt')])

    print("\n" + "-------------"*5)
    print("Total added: {}".format(add_count))
    print("Total deleted: {}".format(delete_count))
    print("Total txt files:", total_files)


    # update embeddings
    # if set the new_files, then will only create embeddings for these new files
    # otherwise, will process all files in the folder, and update the whole vector database
    print("Saving embeddings...")
    if args.rewrite_emb:
        update_embeddings(folder=output_folder, new_files=[])
    else:
        update_embeddings(folder=output_folder, new_files=new_files)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sync papers and embeddings.')
    parser.add_argument('--rewrite_emb', action='store_true',
                        help='Rewrite all embeddings. If not set, only new files will be processed.')
    args = parser.parse_args()

    main(args)