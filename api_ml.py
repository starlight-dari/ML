from flask import Flask, request, jsonify
import os
import json
import subprocess
import threading
import shutil
import io
import base64
import requests
import random
import numpy as np
import cv2
import torch
import boto3
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from PIL import Image
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.metrics import pairwise_distances

# Pinecone & OpenAI
import pinecone
from pinecone import Pinecone
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Diffusers & Segmentation Models
from diffusers import DiffusionPipeline, UNet2DConditionModel
from segment_anything import sam_model_registry, SamPredictor

# AWS
from botocore.client import Config

# .env 파일 로드
load_dotenv()

# AWS 환경변수
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# dreambooth
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 
TRAIN_SCRIPT = "./train_dreambooth.py"

# open_ai & Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 장례식장 정보 JSON 로드
FUNERAL_JSON_PATH = "./funeral_service.json"
def load_json_data(file_path):
    """JSON 파일을 불러와 리스트로 반환"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"JSON 로드 오류: {e}")
        return []
funeral_data = load_json_data(FUNERAL_JSON_PATH)

# Pidinet
# sam_checkpoint = "./sam_vit_b_01ec64.pth"  # SAM checkpoint 파일 경로
# model_type = "vit_b"
# state_dict = torch.load(sam_checkpoint, weights_only=True)
# sam = sam_model_registry[model_type](checkpoint=None)  # Load without unsafe data
# sam.load_state_dict(state_dict)
# predictor = SamPredictor(sam)

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
# Pinecone 인덱스 연결
index_meritz = pc.Index("meritz")
index_samsung = pc.Index("samsung")
index_hanhwa = pc.Index("hanwha")
index_diagnostic = pc.Index("diagnostic")

training_status = {"status": "idle"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def download_s3_images(image_urls, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for i, url in enumerate(image_urls):
        https_url = url  # S3 URL을 HTTPS로 변환
        try:
            response = requests.get(https_url, stream=True)
            if response.status_code == 200:
                save_path = os.path.join(save_dir, f"image_{i}.jpg")
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"✅ Downloaded: {https_url} -> {save_path}")
            else:
                print(f"❌ Failed to download {https_url} (Status Code: {response.status_code})")
        except Exception as e:
            print(f"❌ Error downloading {https_url}: {e}")

def stars_download_s3_images(image_urls, save_folder="./img"):
    """
    HTTP(S) 형식의 S3 이미지 URL 목록을 다운로드하여 로컬 폴더에 저장하는 함수.

    :param image_urls: S3에서 제공하는 이미지의 HTTP URL 리스트
    :param save_folder: 이미지를 저장할 로컬 폴더 (기본값: ./img)
    :return: 저장된 로컬 이미지 경로 리스트
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 리스트가 아니라면 리스트로 변환
    if isinstance(image_urls, str):
        image_urls = [image_urls]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_paths = []
    for image_url in image_urls:
        filename = os.path.basename(image_url)
        save_path = os.path.join(save_folder, filename)

        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # HTTP 에러 처리

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            print(f"✅ 이미지 다운로드 완료: {save_path}")
            saved_paths.append(save_path)

        except requests.exceptions.RequestException as e:
            print(f"❌ 이미지 다운로드 실패: {image_url}, 오류: {e}")
    
    return saved_paths

def upload_svg_to_s3(bucket_name, object_name=None):
    """
    .svg 파일을 S3에 업로드하고 URL을 반환하는 함수
    :param file_path: 로컬 파일 경로
    :param bucket_name: S3 버킷 이름
    :param object_name: S3에 저장할 파일 이름 (기본적으로 로컬 파일 이름과 동일)
    :return: 업로드된 파일의 S3 URL
    """

    try:
        s3_client.upload_file(
            Filename=object_name,
            Bucket=bucket_name, 
            Key=f"test_user/{object_name}",
            ExtraArgs={'ContentType': 'image/png'}  # MIME 타입 지정
        )
        
        # 업로드된 파일의 URL 생성
        file_url = f"https://{bucket_name}.s3.amazonaws.com/test_user/{object_name}"
        return file_url

    except Exception as e:
        print(f"파일 업로드 실패: {e}")
        return None

############################
######### api_rag ##########
############################


def get_embedding(text):
    """텍스트 임베딩 생성 (OpenAI 최신 API 사용)"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 최신 클라이언트 사용
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding  # 변경된 인터페이스
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

def search_index(index, query, top_k=3):
    """주어진 Pinecone 인덱스에서 유사 문서 검색"""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches

    return [
        match["metadata"]["chunk_text"]
        for match in matches if "metadata" in match and "chunk_text" in match["metadata"]
    ]

def generate_answer(query, relevant_texts, prompt_template):
    """LangChain을 사용해 문맥 기반 답변 생성"""
    context = "\n\n".join(relevant_texts)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            api_key=OPENAI_API_KEY
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run({"context": context, "question": query})
    except Exception as e:
        print(f"답변 생성 오류: {e}")
        return "답변을 생성하는 중 오류가 발생했습니다."

@app.route('/rag_get_answer', methods=['POST'])
def get_answer():
    """클라이언트 요청 처리"""
    data = request.json
    route_num = data.get("route_num")
    query = data.get("query")

    if route_num is None or query is None:
        return jsonify({"error": "route_num과 query가 필요합니다."}), 400

    if route_num == 0:
        # 보험 정보 검색 및 답변
        meritz_texts = search_index(index_meritz, query, top_k=3)
        samsung_texts = search_index(index_samsung, query, top_k=3)
        hanhwa_texts = search_index(index_hanhwa, query, top_k=3)
        
        relevant_texts = (
            ["[메리츠]\n" + "\n".join(meritz_texts)] +
            ["[삼성화재]\n" + "\n".join(samsung_texts)] +
            ["[한화]\n" + "\n".join(hanhwa_texts)]
        )

        prompt_template = (
            "당신은 보험 추천 도우미입니다. 아래의 문맥을 참고하여 3개 보험사를 비교하여 질문에 정확하고 이해하기 쉬운 답변을 300자 이내로 해주세요.\n"
            "문맥: {context}\n\n"
            "질문: {question}\n"
            "답변:"
        )
    
    elif route_num == 1:
        # 반려동물 정보 검색 및 답변
        diagnostic_texts = search_index(index_diagnostic, query, top_k=7)

        relevant_texts = diagnostic_texts

        prompt_template = (
            "당신은 노령견 전문 정보 제공 도우미입니다. 아래의 문맥을 참고하여 300자 이내로 질문에 정확하고 이해하기 쉬운 답변을 해주세요.\n"
            "문맥: {context}\n\n"
            "질문: {question}\n"
            "답변:"
        )
    
    elif route_num == 2:
        # 장례식장 정보 검색 및 답변
        relevant_texts = [str(funeral_data)]

        prompt_template = (
            "당신은 장례식장 관련 정보를 제공하는 전문가입니다. "
            "아래의 문맥을 참고하여 사용자가 원하는 서비스를 파악한 후 가장 적합한 장례식장을 문맥에서 찾아 안내해주세요.\n"
            "서울특별시는 경기도와 맞닿아 있고 경기도는 동쪽으로 강원도, 남쪽으로 충청남도, 충청북도와 맞닿아 있음 \
            충청남도와 충청북도 사이에 대전광역시가 있고\
            충청남도 남쪽에 전라북도가 있고\
            충청북도 동남쪽에 경상북도가 있고\
            전라북도와 경상북도 사이 남쪽에 경상남도가 있고\
            전라북도와 경상남도 사이 남서방향에 전라남도가 있음\
            서울특별시는 경기도에 둘러싸여 있고\
            인천광역시는 서울 서쪽에 위치하며 경기도와 접하고 있음\
            대전광역시는 충청남도와 충청북도 사이에 위치하고\
            광주광역시는 전라남도에 위치하며 전라북도와 인접해 있음\
            대구광역시는 경상북도 내에 위치하고\
            부산광역시는 경상남도 남단에 위치하며 남해에 접하고 있음\
            울산광역시는 경상남도 동북쪽에 위치하며 동쪽은 바다(동해)와 접함\
            "
            "문맥: {context}\n\n"
            "질문: {question}\n"
            "답변:"
        )
    
    elif route_num == 3:
        answer = """다음은 펫로스 증후근 극복 프로그램 링크 목록입니다.
마인드카페 센터
: https://center.mindcafe.co.kr/program_petloss

마음치유모임 with 펫로스
: https://www.gangnam.go.kr/contents/mind_healing/1/view.do?mid=ID04_04075401

"""

        return jsonify({"answer": answer})
    
    else:
        return jsonify({"error": "route_num 값이 올바르지 않습니다."}), 400
    answer = generate_answer(query, relevant_texts, prompt_template)
    
    return jsonify({"answer": answer})

############################
## api_letter_dreambooth ###
############################
           
def generate_letter_answer(memories, prompt, openai_api_key):
    if not memories or len(memories) < 2:  # 최소한 성격(character)과 품종(breed)이 있어야 함
        print("❌ 오류: memories 리스트가 너무 짧음.")
        return "기본적인 정보가 부족하여 답변을 생성할 수 없습니다."

    context = "\n\n".join(memories)
    
    full_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "문맥: {context}\n\n"
            "질문: {question}\n"
            "답변:"
        )
    )
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key
        )
        chain = LLMChain(llm=llm, prompt=full_prompt)
        answer = chain.run({"context": context, "question": prompt})
        return answer.strip()
    except Exception as e:
        print(f"❌ 응답 생성 중 오류 발생: {e}")
        return "답변 생성에 실패했습니다."

@app.route('/letter_train', methods=['POST'])
def train_dreambooth():
    data = request.json
    image_urls = data.get("images", [])

    if not image_urls:
        return jsonify({"error": "No images provided"}), 400

    # 🔹 Step 1: Download images from S3
    downloaded_images = download_s3_images(image_urls, "./train_images")

    # 🔹 Step 3: Start training only if all images are available
    command = [
        "accelerate", "launch", "--num_cpu_threads_per_process=1", TRAIN_SCRIPT,
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--instance_data_dir=./train_images",
        "--output_dir=./dreambooth_output",
        "--instance_prompt=a sks pet",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        "--mixed_precision=fp16",
        "--learning_rate=5e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=700",
        "--checkpointing_steps=700"
    ]

    try:
        print("🚀 All images downloaded. Starting training...")
        subprocess.run(command, check=True)  # 🔹 This blocks until training completes
        print("✅ Training completed successfully!")
        return jsonify({"message": "Training completed"}), 200
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return jsonify({"error": "Training failed"}), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status), 200

@app.route('/letter_generate', methods=['POST'])
def generate_images():
    data = request.json
    character = data.get("character", "")
    breed = data.get("breed", "")
    texts = data.get("texts", [])
    memories = [character, breed] + texts

    # GPT로 편지 생성
    letter_prompt = "반려동물의 성격과 반려동물과의 추억을 기록한 게시글을 바탕으로 \
        반려동물이 주인에게 쓰는 따뜻한 편지를 반말로 작성해 주세요."
    letter = generate_letter_answer(memories, letter_prompt, OPENAI_API_KEY )
    
    # GPT로 DreamBooth 프롬프트 추출
    prompt_extraction = "위 내용을 바탕으로 DreamBooth 모델에 적합한 프롬프트를 영어로 아주 짧게 생성하세요.\
        어떤 상황을 묘사하는 내용이며 'a sks ...' 형식으로 시작해야 합니다.\
        (ex) a sks cat on a grass"
    dreambooth_prompt = generate_letter_answer(memories, prompt_extraction, OPENAI_API_KEY )
    dreambooth_prompt = "high quality, J_illustration, " + dreambooth_prompt
    
    print(dreambooth_prompt)
    
    checkpoint_dir = "./dreambooth_output/checkpoint-700"
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(checkpoint_dir, "unet"),
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        unet=unet,
        torch_dtype=torch.float16
    ).to(device)
    
    lora_path = "./J_illustration.safetensors"
    pipeline.load_lora_weights(lora_path)

    # 이미지 생성
    guidance_scales = [5, 6, 7, 8, 9, 10]
    inference_steps = [100]
    generated_images = []
    
    for scale in guidance_scales:
        for step in inference_steps:
            with torch.autocast(device.type):
                result = pipeline(dreambooth_prompt, num_inference_steps=step, guidance_scale=scale)
            generated_images.append(result.images[0])

    # 최종 이미지 6장 선택
    encoded_images = []
    for idx, image in enumerate(generated_images[:6]):
        local_path = f"generated_image_{idx}.png"
        
        # 이미지 저장 (PIL 이미지로 변환하여 PNG 형식으로 저장)
        image.save(local_path, format="PNG")
        print(f"✅ Image saved locally: {local_path}")

        # S3 업로드
        # object_name = f"generated_image_{idx}.png"
        file_url = upload_svg_to_s3(BUCKET_NAME, local_path)  # 로컬 파일 경로 사용

        if file_url:
            encoded_images.append(file_url)
            print(f"✅ Uploaded to S3: {file_url}")
        else:
            print(f"❌ Failed to upload {local_path} to S3")

    shutil.rmtree("./dreambooth_output", ignore_errors=True)
    shutil.rmtree("./train_images", ignore_errors=True)

    return jsonify({"images": encoded_images, "letter": letter})

############################
######### api_stars ########
############################

def process_segmentation(image, mask, edge_threshold1=50, edge_threshold2=150,
                         scale_factor=0.9, mask_threshold=30, max_internal_points=1000):
    """
    1) 마스크에서 가장 큰 윤곽선을 찾아 일정 간격으로 점들을 추출합니다.
    2) 마스크 내부 영역에 대해 Canny 에지 검출 후 에지 점들을 샘플링합니다.
    3) 두 종류의 점(윤곽선, 내부 에지)을 반환합니다.
    """
    if mask.sum() == 0:
        raise ValueError("The mask is empty. Check the input image or segmentation result.")

    # 그레이스케일 변환 (컬러 이미지인 경우)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 마스크 이진화
    _, mask_uint8 = cv2.threshold((mask * 255).astype(np.uint8), mask_threshold, 255, cv2.THRESH_BINARY)
    
    # 윤곽선 추출
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Check the mask content.")

    # 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 윤곽선 근사
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    contour_points = approx_contour[:, 0, :]
    if len(contour_points) < 50:
        contour_points = largest_contour[:, 0, :]

    # 점 샘플링 (너무 많으면 시각화 복잡해질 수 있음)
    step = max(1, len(contour_points) // 60)
    contour_points = contour_points[::step]

    # 내부 영역 마스크 (scale_factor 만큼 줄인 후 중앙 정렬)
    scaled_mask = cv2.resize(mask_uint8, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    dh = (mask.shape[0] - scaled_mask.shape[0]) // 2
    dw = (mask.shape[1] - scaled_mask.shape[1]) // 2
    scaled_mask_padded = cv2.copyMakeBorder(
        scaled_mask,
        top=dh,
        bottom=(mask.shape[0] - scaled_mask.shape[0] - dh),
        left=dw,
        right=(mask.shape[1] - scaled_mask.shape[1] - dw),
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # Canny 엣지 검출
    edges = cv2.Canny(gray_image, edge_threshold1, edge_threshold2)
    # 내부 영역에 해당하지 않는 에지 제거
    edges[scaled_mask_padded == 0] = 0

    # 에지 점 추출
    edge_y, edge_x = np.where(edges > 0)
    edge_points = np.stack((edge_x, edge_y), axis=-1)
    num_points = edge_points.shape[0]
    
    # 최대 max_internal_points 개로 샘플링 (더 많이 뽑히도록 500 -> max_internal_points)
    num_sample = min(num_points, max_internal_points)
    if num_points > 0:
        random_indices = np.random.choice(num_points, num_sample, replace=False)
        random_edge_points = edge_points[random_indices]
    else:
        random_edge_points = np.empty((0, 2), dtype=int)

    return contour_points, random_edge_points
    
def sample_pidinet_edges(pidinet_output, mask, scale_factor=0.9, sample_size=1000):
    """
    pidinet 결과에서 흰색(전경) 부분을 샘플링하는 함수
    1) 마스크에서 가장 큰 윤곽선을 찾아 일정 간격으로 점들을 추출
    2) pidinet 출력에서 마스크 내부의 흰색(전경) 점들을 샘플링
    """
    if mask.sum() == 0:
        raise ValueError("The mask is empty. Check the input image or segmentation result.")
    
    # pidinet_output을 numpy 배열로 변환
    if not isinstance(pidinet_output, np.ndarray):
        pidinet_output = np.array(pidinet_output.convert("L"))  # 흑백 변환
    
    # 마스크 이진화
    _, mask_uint8 = cv2.threshold((mask * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    
    # 윤곽선 추출
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Check the mask content.")
    
    # 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 윤곽선 근사화 및 샘플링
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    contour_points = approx_contour[:, 0, :]
    
    if len(contour_points) < 50:
        contour_points = largest_contour[:, 0, :]
    
    step = max(1, len(contour_points) // 60)
    contour_points = contour_points[::step]
    
    # 내부 영역 마스크(scale_factor 만큼 축소)
    scaled_mask = cv2.resize(mask_uint8, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    dh = (mask.shape[0] - scaled_mask.shape[0]) // 2
    dw = (mask.shape[1] - scaled_mask.shape[1]) // 2
    scaled_mask_padded = cv2.copyMakeBorder(
        scaled_mask, dh, mask.shape[0] - scaled_mask.shape[0] - dh, 
        dw, mask.shape[1] - scaled_mask.shape[1] - dw, 
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    # 기본 threshold 값 및 설정
    base_threshold = 0
    min_threshold = 5
    max_threshold = 40
    target_ratio = 0.05  # 목표 흰색 비율
    scaling_factor = 10  # 변화량 조절

    # 전체 픽셀에서 흰색 비율 계산
    total_pixels = np.sum(scaled_mask_padded > 0)  # 마스크 내부의 유효 픽셀 개수
    edge_pixels = np.sum(pidinet_output > base_threshold)  # 초기 threshold에서 검출된 픽셀 개수
    white_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0  # 흰색 비율

    # 선형 스케일링을 활용한 동적 threshold 계산
    dynamic_threshold = base_threshold + scaling_factor * (white_ratio - target_ratio)
    # dynamic_threshold = np.clip(dynamic_threshold, min_threshold, max_threshold)  # min~max 제한/
    dynamic_threshold = 15

    # 새로운 threshold로 흰색 픽셀 찾기
    edge_y, edge_x = np.where((pidinet_output > dynamic_threshold) & (scaled_mask_padded > 0))
    edge_points = np.stack((edge_x, edge_y), axis=-1)

    print(f"White Ratio: {white_ratio:.4f}, Adjusted Threshold: {dynamic_threshold:.2f}")
    
    # 샘플링 수행 (마스크 내부에서 선택)
    num_points = edge_points.shape[0]
    num_sample = min(num_points, sample_size)
    if num_points > 0:
        random_indices = np.random.choice(num_points, num_sample, replace=False)
        random_edge_points = edge_points[random_indices]
    else:
        random_edge_points = np.empty((0, 2), dtype=int)
    
    return contour_points, random_edge_points

def load_and_preprocess_image(image_path):
    """이미지를 불러오고 전처리하는 함수"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러오지 못했습니다: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

def run_sam_segmentation(image_rgb, predictor, point):
    """SAM 모델을 이용해 마스크 예측 수행"""
    predictor.set_image(image_rgb)
    
    h, w, _ = image_rgb.shape
    point = np.array([point])
    input_label = np.array([1])  # 1: object

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask_index = int(np.argmax(scores))
    return masks[best_mask_index]

def extract_and_sort_centroids(contour_points, n_clusters, n_points):
    """
    주어진 컨투어 포인트에서 K-Means 클러스터링을 사용하여 
    n_points 개의 중심점을 랜덤하게 선택한 후, 가장 가까운 점들로 정렬하는 함수.

    :param contour_points: (N, 2) 형태의 numpy 배열, 컨투어 좌표 리스트
    :param n_clusters: 클러스터 개수
    :param n_points: 선택할 중심점 개수
    :return: 최근접 순서로 정렬된 중심점 리스트
    """
    if len(contour_points) < n_clusters:
        raise ValueError("Contour points 개수가 클러스터 개수보다 많아야 합니다.")

    # K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(contour_points)

    # 중심점(centroids) 가져오기
    centroids = kmeans.cluster_centers_

    # 중심점 중 n_points 개 랜덤 선택
    num_selected = min(n_points, n_clusters)
    selected_centroids = np.array(random.sample(centroids.tolist(), num_selected))

    # 최근접 이웃 방식으로 정렬
    sorted_centroids = nearest_neighbor_sort(selected_centroids)

    return sorted_centroids

def nearest_neighbor_sort(points):
    """
    최근접 이웃(Nearest Neighbor) 알고리즘을 사용하여 점들을 정렬하는 함수.

    :param points: (N, 2) 형태의 numpy 배열, 랜덤 선택된 중심점 리스트
    :return: 최근접 순서로 정렬된 numpy 배열
    """
    points = points.tolist()
    sorted_points = [points.pop(0)]  # 첫 번째 점을 시작점으로 설정

    while points:
        last_point = sorted_points[-1]
        nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(last_point)))
        sorted_points.append(nearest_point)
        points.remove(nearest_point)

    return np.array(sorted_points)

def generate_mst_graph(major_points):
    """MST를 생성하고 연결을 보장하는 함수"""
    knn_graph = kneighbors_graph(major_points, n_neighbors=3, mode='distance')
    mst_matrix = minimum_spanning_tree(knn_graph)
    coo = mst_matrix.tocoo()
    edges = np.vstack((coo.row, coo.col)).T

    n_components, labels = connected_components(mst_matrix)
    if n_components > 1:
        distance_matrix = pairwise_distances(major_points)
        added_edges = []

        for i in range(n_components - 1):
            for j in range(i + 1, n_components):
                mask_i, mask_j = labels == i, labels == j
                min_dist, min_edge = np.inf, None

                for idx_i in np.where(mask_i)[0]:
                    for idx_j in np.where(mask_j)[0]:
                        if distance_matrix[idx_i, idx_j] < min_dist:
                            min_dist = distance_matrix[idx_i, idx_j]
                            min_edge = (idx_i, idx_j)

                if min_edge:
                    added_edges.append(min_edge)

        edges = np.vstack((edges, added_edges))
    
    return edges

def image_to_svg(image, svg_path, threshold=128):
    """이미지를 SVG 포맷으로 변환"""
    height, width = image.shape
    dwg = svgwrite.Drawing(svg_path, size=(width, height))

    for y in range(height):
        for x in range(width):
            if image[y, x] >= threshold:
                dwg.add(dwg.circle(center=(x, y), r=0.5, fill='gray', stroke='none'))
    
    dwg.save()

def enhance_contrast(image):
    """Convert image to grayscale and enhance contrast using CLAHE"""
    if len(image.shape) == 3:  # Convert to grayscale if the image has multiple channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure the image is 8-bit (uint8)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

@app.route("/stars_run_pidinet", methods=["POST"])
def run_pidinet():
    """
    클라이언트로부터 이미지 리스트를 받아 PiDiNet 실행.
    """
    data = request.json
    print(f"📥 Received Data: {data}")  # Debugging log

    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No images provided"}), 400

    # 단일 문자열이면 리스트로 변환
    if isinstance(image_url, str):
        image_url = [image_url]
        
    stars_download_s3_images(image_urls = image_url, save_folder="./img")
    
    # 🔹 Step 3: PiDiNet 실행 명령어
    command = [
        "python", "pidinet-master/main.py",
        "--model", "pidinet_converted",
        "--config", "carv4",
        "--sa", "--dil",
        "-j", "4",
        "--gpu", "0",
        "--resume",
        "--savedir", "./img_edges",
        "--datadir", "./img",
        "--dataset", "Custom",
        "--evaluate", "./table5_pidinet.pth",
        "--evaluate-converted"
    ]

    try:
        print("🚀 PiDiNet 실행 중...")
        result = subprocess.run(command, check=True)  # 실행 (완료될 때까지 대기)
        print("✅ PiDiNet 실행 완료!")

        return jsonify({
            "message": "PiDiNet execution completed",
        }), 200

    except subprocess.CalledProcessError as e:
        print(f"❌ PiDiNet 실행 실패: {e}")
        return jsonify({"error": "PiDiNet execution failed"}), 500

@app.route("/stars_process_image", methods=["POST"])
def process_image():
    """
    클라이언트로부터 이미지 URL과 관련 정보를 받아 처리.
    """
    data = request.json
    image_url = data.get("image_url")
    point = data.get("point")

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_name = os.path.basename(image_url)
    image_path = "./img/" + image_name
    
    image, image_rgb = load_and_preprocess_image(image_path)
    mask = run_sam_segmentation(image_rgb, predictor, point)
    
    mask_ratio = np.count_nonzero(mask) / mask.size
    max_internal_points = int(mask_ratio * 3000)
    
    contour_points2, internal_points2 = process_segmentation(image, mask, max_internal_points=max_internal_points)

    image_path = f"./img_edges/eval_results/imgs_epoch_019/{os.path.splitext(image_name)[0]}.png"
    image = Image.open(image_path)
    image = np.array(image.convert("L"))
    
    masked_image = np.zeros_like(image, dtype=np.float32) 
    masked_image[mask > 0] = image[mask > 0] 
    processed_image = enhance_contrast(masked_image)
    
    svg_path = f"{os.path.splitext(image_name)[0]}.svg"
    png_path = f"{os.path.splitext(image_name)[0]}_masked.png"
    
    image_to_svg(processed_image, svg_path)
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    
    try:
        svg_path = upload_svg_to_s3(BUCKET_NAME,f"{os.path.splitext(image_name)[0]}.svg")
    except Exception as e:
        print(f"파일 업로드 실패: {e}")
    
    contour_points1, internal_points1 = sample_pidinet_edges(image, mask, sample_size=max_internal_points)
    
    internal_points = np.vstack((internal_points1, internal_points2))
    contour_points = np.vstack((contour_points2))

    major_contour = extract_and_sort_centroids(contour_points, n_clusters=20, n_points=5)
    major_internal = extract_and_sort_centroids(internal_points, n_clusters=20, n_points=10)
    
    major_points = np.vstack((major_contour, major_internal))
    edges = generate_mst_graph(major_points)

    return jsonify({
        "svg_path": svg_path,
        "edges": edges.tolist(),  # Convert NumPy arrays to lists
        "major_points": major_points.astype(int).tolist()
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)