import os
import base64
import numpy as np
import json
from datetime import datetime, timedelta
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from openai import OpenAI
from dotenv import load_dotenv

# Configuração da página
st.set_page_config(
    page_title="Análise Inteligente de Documentos",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentAnalyzer:
    def __init__(self):
        self.client = self._initialize_openai()
        
    def _initialize_openai(self):
        """Inicializa o cliente OpenAI"""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Chave da API OpenAI não encontrada!")
            st.stop()
        return OpenAI(api_key=api_key)
    
    def _process_value(self, value_str):
        """Processa o valor para garantir formato decimal correto"""
        try:
            # Remove caracteres não numéricos exceto vírgula e ponto
            clean_value = ''.join(c for c in value_str if c.isdigit() or c in '.,')
            
            # Se não tem separador decimal, verifica o tamanho
            if ',' not in clean_value and '.' not in clean_value:
                # Se for maior que 2 dígitos, assume que os últimos 2 são centavos
                if len(clean_value) > 2:
                    clean_value = clean_value[:-2] + '.' + clean_value[-2:]
            else:
                # Substitui vírgula por ponto
                clean_value = clean_value.replace(',', '.')
                
            # Converte para float
            value = float(clean_value)
            
            # Se o valor parece muito alto para um cupom comum, divide por 100
            if value > 1000:  # você pode ajustar este limite
                value = value / 100
                
            return value
        except:
            return None
    
    def analyze_document(self, file):
        """Analisa o documento usando GPT-4o-mini"""
        try:
            bytes_data = file.getvalue()
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extraia as seguintes informações do documento e retorne em JSON:
                                {
                                    "tipo_documento": "tipo do documento (nota fiscal, recibo, etc)",
                                    "valor_total": "valor numérico (mantenha vírgulas e pontos)",
                                    "data": "YYYY-MM-DD",
                                    "estabelecimento": "nome do estabelecimento",
                                    "categoria": "categoria do gasto",
                                    "metodo_pagamento": "forma de pagamento"
                                }"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            try:
                json_result = json.loads(result)
                # Processa o valor total
                if 'valor_total' in json_result:
                    json_result['valor_total'] = self._process_value(str(json_result['valor_total']))
                return json_result
            except json.JSONDecodeError:
                # Se falhar ao decodificar JSON, tenta extrair informações do texto
                extracted_info = self._extract_info_from_text(result)
                if extracted_info and 'valor_total' in extracted_info:
                    extracted_info['valor_total'] = self._process_value(str(extracted_info['valor_total']))
                return extracted_info
                
        except Exception as e:
            st.error(f"Erro ao processar documento: {str(e)}")
            return None
    
    def _extract_info_from_text(self, text):
        """Extrai informações do texto quando o JSON falha"""
        try:
            # Tenta encontrar padrões no texto
            info = {
                "tipo_documento": "",
                "valor_total": "",
                "data": "",
                "estabelecimento": "",
                "categoria": "",
                "metodo_pagamento": ""
            }
            
            # Procura por linhas que contenham as chaves
            lines = text.split('\n')
            for line in lines:
                line = line.lower().strip()
                if "tipo" in line:
                    info["tipo_documento"] = line.split(":")[-1].strip()
                elif "valor" in line:
                    value = line.split(":")[-1].strip()
                    # Remove R$ e outros caracteres
                    value = ''.join(filter(str.isdigit, value))
                    info["valor_total"] = value
                elif "data" in line:
                    info["data"] = line.split(":")[-1].strip()
                elif "estabelecimento" in line:
                    info["estabelecimento"] = line.split(":")[-1].strip()
                elif "categoria" in line:
                    info["categoria"] = line.split(":")[-1].strip()
                elif "pagamento" in line:
                    info["metodo_pagamento"] = line.split(":")[-1].strip()
            
            return info
            
        except Exception as e:
            st.error(f"Erro ao extrair informações: {str(e)}")
            return None

# Interface principal
st.title("📊 Análise Inteligente de Documentos")
st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
if 'documents_data' not in st.session_state:
    st.session_state.documents_data = pd.DataFrame(
        columns=['tipo_documento', 'valor_total', 'data', 'estabelecimento', 
                'categoria', 'metodo_pagamento', 'nome_arquivo']
    )

# Inicialização do analisador
analyzer = DocumentAnalyzer()

# Upload de arquivos
uploaded_files = st.file_uploader(
    "Faça upload dos seus documentos",
    type=['jpg'],
    accept_multiple_files=True
)

# Processamento dos arquivos
if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processando {file.name}...")
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        
        if not st.session_state.documents_data['nome_arquivo'].str.contains(file.name).any():
            result = analyzer.analyze_document(file)
            
            if result:
                result['nome_arquivo'] = file.name
                new_data = pd.DataFrame([result])
                st.session_state.documents_data = pd.concat(
                    [st.session_state.documents_data, new_data], 
                    ignore_index=True
                )
    
    status_text.text("Processamento concluído!")
    progress_bar.empty()

# Exibição dos dados processados
if not st.session_state.documents_data.empty:
    st.subheader("📋 Documentos Processados")
    
    # Configuração do AgGrid
    gb = GridOptionsBuilder.from_dataframe(st.session_state.documents_data)
    gb.configure_default_column(
        editable=True,
        filterable=True,
        sortable=True
    )
    gb.configure_selection('multiple', use_checkbox=True)
    
    grid_response = AgGrid(
        st.session_state.documents_data,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=400
    )
    
    # Análises
    try:
        st.subheader("📈 Análises")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_gasto = pd.to_numeric(st.session_state.documents_data['valor_total'], errors='coerce').sum()
            st.metric("Total Gasto", f"R$ {total_gasto:,.2f}")
        
        with col2:
            media_gasto = pd.to_numeric(st.session_state.documents_data['valor_total'], errors='coerce').mean()
            st.metric("Média por Documento", f"R$ {media_gasto:,.2f}")
        
        with col3:
            num_docs = len(st.session_state.documents_data)
            st.metric("Total de Documentos", num_docs)
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            df_plot = st.session_state.documents_data.copy()
            df_plot['valor_total'] = pd.to_numeric(df_plot['valor_total'], errors='coerce')
            
            fig_categoria = px.pie(
                df_plot,
                names='categoria',
                values='valor_total',
                title='Gastos por Categoria'
            )
            st.plotly_chart(fig_categoria, use_container_width=True)
        
        with col2:
            df_plot['data'] = pd.to_datetime(df_plot['data'], errors='coerce')
            df_plot = df_plot.sort_values('data')
            
            fig_timeline = px.line(
                df_plot,
                x='data',
                y='valor_total',
                title='Evolução dos Gastos'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro ao gerar análises: {str(e)}")

    # Exportação
    st.sidebar.subheader("📥 Exportar Dados")
    
    csv = st.session_state.documents_data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Baixar CSV",
        csv,
        "documentos_analisados.csv",
        "text/csv"
    )

    # Opção para exportar Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        st.session_state.documents_data.to_excel(writer, index=False)
    
    st.sidebar.download_button(
        label="📊 Baixar Excel",
        data=buffer.getvalue(),
        file_name='analise_documentos.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

else:
    st.info("👆 Faça upload de documentos para começar!")

# Opção para limpar dados
if st.sidebar.button("🗑️ Limpar Dados"):
    st.session_state.documents_data = pd.DataFrame(
        columns=['tipo_documento', 'valor_total', 'data', 'estabelecimento', 
                'categoria', 'metodo_pagamento', 'nome_arquivo']
    )
    st.experimental_rerun()

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Desenvolvido Union It </p>
    </div>
    """, 
    unsafe_allow_html=True
)
