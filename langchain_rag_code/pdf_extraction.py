from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

def separate_elements(chunks):
    tables, texts = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    return tables, texts

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64
