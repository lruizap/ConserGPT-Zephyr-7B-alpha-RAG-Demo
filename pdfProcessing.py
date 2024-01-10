import re
import fitz  # PyMuPDF
import io
import os
import PyPDF2
import glob


def quitar_cabecera_pie(pdf_path, output_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_writer = PyPDF2.PdfWriter()

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            media_box = page.mediabox
            page_width = media_box.width
            page_height = media_box.height

            # Puedes ajustar estos valores según tus necesidades
            margin_top = 105
            margin_bottom = 50

            # Eliminar cabecera
            media_box.lower_left = (0, margin_bottom)
            media_box.upper_right = (page_width, page_height)

            # Eliminar pie de página
            media_box.lower_left = (0, 0)
            media_box.upper_right = (page_width, page_height - margin_top)

            pdf_writer.add_page(page)

        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)


def cortar_pdf_desde_anexo(archivo_pdf, palabra_clave="ANEXO"):
    doc = fitz.open(archivo_pdf)

    # Buscar la página que contiene la palabra clave "ANEXO"
    indice_anexo = None
    for num_pagina in range(doc.page_count):
        pagina = doc[num_pagina]
        texto_pagina = pagina.get_text()

        if palabra_clave in texto_pagina:
            indice_anexo = num_pagina
            break

    if indice_anexo is not None:
        # Guardar las páginas antes del anexo
        doc_antes_anexo = fitz.open()
        doc_antes_anexo.insert_pdf(doc, from_page=0, to_page=indice_anexo-1)
        doc_antes_anexo.save(f"{archivo_pdf}")
        doc_antes_anexo.close()

    doc.close()


def limpiar_y_convertir_a_txt(input_path, output_txt_path):
    doc = fitz.open(input_path)
    texto_completo = ""

    for pagina_num in range(doc.page_count):
        pagina = doc[pagina_num]
        lineas_pagina = pagina.get_text("text").splitlines()

        texto_pagina = "\n".join(lineas_pagina)

        texto_completo += texto_pagina

    # Guardar el texto en un archivo de texto
    with io.open(output_txt_path, "w", encoding="utf-8") as archivo_txt:
        archivo_txt.write(texto_completo)

    doc.close()


def procesar_archivo(input_txt_path, output_txt_path):
    with open(input_txt_path, 'r', encoding='utf-8') as archivo_entrada:
        lineas = archivo_entrada.readlines()

    # Eliminar la primera línea
    lineas = lineas[1:]

    # Convertir el contenido del archivo a una sola línea
    contenido_linea = " ".join(lineas)

    contenido_linea = re.sub(r'\n', r'', contenido_linea)
    contenido_linea = re.sub(r'\s\s', r'', contenido_linea)
    contenido_linea = re.sub(r'(\w)-\s*(\w)', r'\1\2', contenido_linea)

    # Eliminar el contenido entre "FIRMADO POR" y "Es copia auténtica de documento electrónico"
    contenido_linea = re.sub(
        r'FIRMADO POR.*?Es copia auténtica de documento electrónico', '', contenido_linea, flags=re.DOTALL)

    # Guardar el contenido procesado en un nuevo archivo de texto
    with open(output_txt_path, 'w', encoding='utf-8') as archivo_salida:
        archivo_salida.write(contenido_linea)


def procesar_carpeta(carpeta, carpeta_destino):
    # Obtener la lista de archivos PDF en la carpeta
    archivos_pdf = glob.glob(os.path.join(carpeta, '*.pdf'))

    for archivo_pdf in archivos_pdf:
        # Crear los nombres de archivo de salida
        nombre_base = os.path.splitext(os.path.basename(archivo_pdf))[0]
        path_txt = os.path.join(carpeta_destino, f'{nombre_base}.txt')

        # Ejecutar la función de procesamiento para cada archivo PDF
        archivo_pdf = archivo_pdf.replace('\\', '/')
        path_txt = path_txt.replace('\\', '/')
        processing_pdf(archivo_pdf, path_txt)


def processing_pdf(path, path_txt):
    quitar_cabecera_pie(path, path)
    cortar_pdf_desde_anexo(path)
    limpiar_y_convertir_a_txt(path, path_txt)
    procesar_archivo(path_txt, path_txt)


# Llamada a la función con la ruta de la carpeta que deseas procesar
carpeta_a_procesar = './pdf_folder/pdf'
carpeta_destino = './pdf_folder/txt'
procesar_carpeta(carpeta_a_procesar, carpeta_destino)
