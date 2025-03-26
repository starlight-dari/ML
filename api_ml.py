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
from peft import PeftModel
from diffusers import StableDiffusionPipeline
import gc

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
from ultralytics import YOLO, SAM

# AWS
from botocore.client import Config

import datetime
from PIL import ImageStat

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

training_status = {"status": "idle"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAM

# sam_checkpoint = "./sam_vit_b_01ec64.pth"  # SAM checkpoint 파일 경로
# # 모델 파일 경로 검증 (파일이 실제 존재하는지 확인)
# if not os.path.isfile(sam_checkpoint):
#     raise FileNotFoundError(f"Model checkpoint file '{sam_checkpoint}' not found.")
# model_type = "vit_b"
# state_dict = torch.load(sam_checkpoint, weights_only=True)
# sam = sam_model_registry[model_type](checkpoint=None)  # Load without unsafe data
# sam.load_state_dict(state_dict)
# predictor = SamPredictor(sam)

yolo_model = YOLO("yolov8x.pt")
sam_model = SAM("sam_b.pt")

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
# Pinecone 인덱스 연결
index_meritz = pc.Index("meritz")
index_samsung = pc.Index("samsung")
index_hanhwa = pc.Index("hanwha")
index_diagnostic = pc.Index("diagnostic")

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
            ExtraArgs={'ContentType': 'image/svg+xml'}  # 올바른 SVG MIME 타입
        )
        
        # 업로드된 파일의 URL 생성
        file_url = f"https://{bucket_name}.s3.amazonaws.com/test_user/{object_name}"
        return file_url

    except Exception as e:
        print(f"파일 업로드 실패: {e}")
        return None

def upload_png_to_s3(bucket_name, object_name, pet_id):
    """
    .png 파일을 S3에 업로드하고 URL을 반환하는 함수
    :param bucket_name: S3 버킷 이름
    :param object_name: S3에 저장할 파일 이름
    :param pet_id: 반려동물 ID
    :return: 업로드된 파일의 S3 URL
    """
    try:
        # 현재 날짜 가져오기 (YYYYMMDD 형식)
        current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # S3에 업로드할 키 경로 설정 (파일명 뒤에 날짜 추가)
        object_name_with_date = f"{object_name}_{current_date}.png"
        s3_key = f"letters/{pet_id}/{object_name_with_date}"

        # 파일 업로드
        s3_client.upload_file(
            Filename=object_name,
            Bucket=bucket_name, 
            Key=s3_key,
            ExtraArgs={'ContentType': 'image/png'}  
        )
        
        # 업로드된 파일의 정확한 URL 생성
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
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

def search_index(index, query, top_k=3, score_threshold=0.45):
    """주어진 Pinecone 인덱스에서 유사 문서 검색 (유사도 필터링 추가)"""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, include_values=False)
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches

    # 유사도 필터링 추가
    filtered_chunks = [
        match["metadata"]["chunk_text"]
        for match in matches
        if "metadata" in match and "chunk_text" in match["metadata"]
        and match.get("score", 0) >= score_threshold
    ]

    return filtered_chunks


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
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON request"}), 400

        route_num = data.get("route_num")
        query = data.get("query")
        
        if route_num is None or query is None:
            return jsonify({"error": "route_num과 query가 필요합니다."}), 400

        if route_num == 0:
            meritz_texts = search_index(index_meritz, query, top_k=3)
            samsung_texts = search_index(index_samsung, query, top_k=3)
            hanhwa_texts = search_index(index_hanhwa, query, top_k=3)
            
            if not any([meritz_texts, samsung_texts, hanhwa_texts]):
                return jsonify({"answer": "저는 보험 관련 내용만 답변해드릴 수 있습니다."})
            
            relevant_texts = (
                ["[메리츠]\n" + "\n".join(meritz_texts)] +
                ["[삼성화재]\n" + "\n".join(samsung_texts)] +
                ["[한화]\n" + "\n".join(hanhwa_texts)]
            )

            prompt_template = (
                "당신은 보험 추천 도우미입니다. 아래의 문맥을 참고하여 3개 보험사를 비교하여 질문에 정확하고 이해하기 쉬운 답변을 300자 이내의 완결된 문장으로 답변해주세요.\n"
                "문맥: {context}\n\n"
                "질문: {question}\n"
                "답변:"
            )
        
        elif route_num == 1:
            diagnostic_texts = search_index(index_diagnostic, query, top_k=7)
            
            if not diagnostic_texts:
                return jsonify({"answer": "저는 노령견 관련 정보만 답변해드릴 수 있습니다."})
            
            relevant_texts = diagnostic_texts

            prompt_template = (
                "당신은 노령견 전문 정보 제공 도우미입니다. 아래의 문맥을 참고하여 질문에 정확하고 이해하기 쉬운 답변을 300자 이내의 완결된 문장으로 답변해주세요.\n"
                "문맥: {context}\n\n"
                "질문: {question}\n"
                "답변:"
            )
        
        elif route_num == 2:
            relevant_texts = [str(funeral_data)]

            prompt_template = (
                "당신은 장례식장 관련 정보를 제공하는 전문가입니다. "
                "아래의 문맥을 참고하여 사용자가 원하는 서비스를 파악한 후 가장 적합한 장례식장을 문맥에서 찾아 300자 이내의 완결된 문장으로 답변해 안내해주세요.\n"
                "문맥: {context}\n\n"
                "질문: {question}\n"
                "답변:"
            )
        
        elif route_num == 3:
            answer = """다음은 펫로스 증후군 극복 프로그램 링크 목록입니다.\n마인드카페 센터\n: https://center.mindcafe.co.kr/program_petloss\n\n마음치유모임 with 펫로스\n: https://www.gangnam.go.kr/contents/mind_healing/1/view.do?mid=ID04_04075401\n"""
            return jsonify({"answer": answer})
        
        else:
            return jsonify({"error": "route_num 값이 올바르지 않습니다."}), 400
        
        try:
            answer = generate_answer(query, relevant_texts, prompt_template)
        except Exception as e:
            print(f"❌ GPT 답변 생성 실패: {e}")
            return jsonify({"error": "Failed to generate answer"}), 500
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

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

def is_black_image(image, threshold=10):
    """이미지가 거의 검정인지 확인하는 함수"""
    grayscale = image.convert("L")  # 흑백으로 변환
    stat = ImageStat.Stat(grayscale)
    avg_brightness = stat.mean[0]
    return avg_brightness < threshold  # 밝기 평균이 threshold보다 작으면 검정으로 판단

def generate_dreambooth(dreambooth_prompt, pet_id):
    checkpoint_dir = "./dreambooth_output/checkpoint-450"
    
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

    max_guidance_scale = 13 # 가장 큰 값 사용
    inference_steps = 100  # 고정된 스텝 수
    generated_images = []

    for _ in range(6):
        result = pipeline(dreambooth_prompt, num_inference_steps=inference_steps, guidance_scale=max_guidance_scale)
        generated_images.append(result.images[0])

    # 최종 이미지 6장 선택
    encoded_images = []
    for idx, image in enumerate(generated_images[:6]):
        if is_black_image(image):
            print(f"⚠️ Skipped black image at index {idx}")
            continue

        local_path = f"generated_image_{idx}.png"

        # 이미지 저장
        image.save(local_path, format="PNG")
        print(f"✅ Image saved locally: {local_path}")

        # S3 업로드
        file_url = upload_png_to_s3(BUCKET_NAME, local_path, pet_id)

        if file_url:
            encoded_images.append(file_url)
            print(f"✅ Uploaded to S3: {file_url}")
        else:
            print(f"❌ Failed to upload {local_path} to S3")

    # 불필요한 파일 삭제
    shutil.rmtree("./dreambooth_output", ignore_errors=True)
    shutil.rmtree("./train_images", ignore_errors=True)

    # GPU 메모리 정리
    del pipeline
    del unet
    pipeline = None
    unet = None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    gc.collect()  # 추가 호출
    torch.cuda.ipc_collect()
    
    return encoded_images

@app.route('/letter_train', methods=['POST'])
def train_dreambooth():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON request"}), 400

        image_urls = data.get("images", [])
        if not image_urls:
            return jsonify({"error": "No images provided"}), 400

        image_urls *= 2  # 데이터 증강
        
        download_s3_images(image_urls, "./train_images")

        command = [
            "accelerate", "launch", "--num_cpu_threads_per_process=4", TRAIN_SCRIPT,
            "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
            "--instance_data_dir=./train_images",
            "--output_dir=./dreambooth_output",
            "--instance_prompt=a sks pet",
            "--resolution=256",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            "--gradient_checkpointing",
            "--mixed_precision=fp16",
            "--learning_rate=5e-6",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--max_train_steps=450",
            "--checkpointing_steps=450",
            "--enable_xformers_memory_efficient_attention",
            "--use_8bit_adam",
        ]

        def run_training():
            global training_status
            training_status["status"] = "running"
            try:
                print("🚀 Training started in background...")
                subprocess.run(command, check=True)
                print("✅ Training completed successfully!")
                training_status["status"] = "completed"
            except subprocess.CalledProcessError as e:
                print(f"❌ Training failed: {e}")
                if "CUDA out of memory" in str(e) or "GPU" in str(e):
                    print("⚠️ GPU 메모리 부족 오류 발생. 모델 훈련을 중지합니다.")
                    training_status["status"] = "failed - GPU memory issue"
                else:
                    training_status["status"] = "failed"
            except Exception as e:
                print(f"⚠️ Unexpected error during training: {e}")
                training_status["status"] = "failed"

        # 새로운 쓰레드에서 훈련 실행 (비동기 처리)
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        return jsonify({"message": "Training started"}), 202  # 202 Accepted: 비동기 요청
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status), 200

@app.route('/letter_generate', methods=['POST'])
def generate_images():
    
    # pipeline과 unet을 None으로 설정
    global pipeline, unet
    pipeline = None
    unet = None
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON request"}), 400

    try:
        character = data.get("character", "")
        breed = data.get("breed", "")
        texts = data.get("texts", [])
        pet_id = int(data.get("pet_id", 0))
        pet_name = data.get("pet_name", "")
        member_name = data.get("member_name", "")
        nickname = data.get("nickname", "")
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400

    memories = [character, breed] + texts

    # GPT로 편지 생성
    try:
        letter_prompt = f"반려동물의 성격과 종, 반려동물과의 추억을 기록한 게시글을 바탕으로 반려동물이 사후 하늘나라에서 주인에게 쓰는 편지를 작성해 주세요. 따뜻하고 감성적인 느낌으로 작성해주세요. 말투는 반말로 해주세요. 반려동물의 이름은 {pet_name}입니다. 주인은 {nickname}의 호칭으로 불러주세요."
        letter = generate_letter_answer(memories, letter_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    try:
        title_prompt = f"이 편지 내용의 제목을 10자 이내로 작성해주세요."
        title = generate_letter_answer(letter, title_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    title = "💌 " + title
    
    print("⭐ title: ", title)

    # GPT로 DreamBooth 프롬프트 추출
    try:
        prompt_extraction = f"위 내용을 바탕으로 DreamBooth 모델에 적합한 프롬프트를 영어로 아주 짧게 생성하세요. 생성 대상인 반려동물 종류는 {breed}이고 'a sks ...' 형식으로 시작해야 합니다."
        dreambooth_prompt = generate_letter_answer(memories, prompt_extraction, OPENAI_API_KEY)
        dreambooth_prompt = "high quality, J_illustration, " + dreambooth_prompt
    except Exception as e:
        return jsonify({"error": f"Prompt extraction failed: {e}"}), 500
    
    print("🎈 dreambooth_prompt :", dreambooth_prompt)

    try:
        encoded_images = generate_dreambooth(dreambooth_prompt, pet_id)
    except Exception as e:
        return jsonify({"error": f"Dreambooth generation failed: {e}"}), 500
    
    return jsonify({"images": encoded_images, "letter": letter, "title": title})

@app.route('/letter_generate_random', methods=['POST'])
def generate_images_random():
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON request"}), 400

    try:
        character = data.get("character", "")
        breed = data.get("breed", "")
        pet_name = data.get("pet_name", "")
        member_name = data.get("member_name", "")
        nickname = data.get("nickname", "")
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400

    memories = [character, breed]

    # GPT로 편지 생성
    try:
        letter_topics = [
            "무지개 다리 건너에서의 나날들", "너와 함께했던 가장 행복한 순간", "처음 너를 만났을 때의 기억",
            "내가 가장 좋아했던 음식과 간식", "내가 가장 좋아했던 장소", "우리만의 특별한 의식",
            "네가 해준 최고의 보살핌", "내가 가끔 사고를 쳤을 때의 이야기", "너를 보며 느꼈던 따뜻한 감정",
            "나를 처음 불렀을 때의 기억", "나를 처음 쓰다듬었을 때의 기분", "네가 나를 위해 해준 가장 특별한 일"
        ]
        letter_topic = random.choice(letter_topics)
        memories = [character, breed] + [letter_topic]

        letter_prompt = f"반려동물의 성격과 종, 반려동물과의 추억을 기록한 게시글을 바탕으로 {letter_topic}을 주제로 반려동물이 사후 하늘나라에서 주인에게 쓰는 안부 인사 편지를 작성해 주세요. 따뜻하고 감성적은 문체로 작성해 주주세요. 말투는 반말로 작성해 주세요. 반려동물의 이름은 {pet_name}입니다. 주인은 {nickname}의 호칭으로 불러주세요."
        letter = generate_letter_answer(memories, letter_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    try:
        title_prompt = f"이 편지 내용의 제목을 10자 이내로 작성해주세요."
        title = generate_letter_answer(letter, title_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    title = "💌 " + title
    
    print("⭐ title: ", title)
    
    return jsonify({"letter": letter, "title": title})

@app.route('/letter_generate_birth_death', methods=['POST'])
def generate_images_birth_death():
    # pipeline과 unet을 None으로 설정
    global pipeline, unet
    pipeline = None
    unet = None
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON request"}), 400

    try:
        character = data.get("character", "")
        breed = data.get("breed", "")
        texts = data.get("texts", []) # 생일, 사망일
        pet_id = int(data.get("pet_id", 0))
        pet_name = data.get("pet_name", "")
        member_name = data.get("member_name", "")
        nickname = data.get("nickname", "")
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400

    memories = [character, breed] + texts

    # GPT로 편지 생성
    try:
        letter_prompt = f"오늘은 특별한 날입니다. 반려동물의 성격과 종, 반려동물과의 추억을 기록한 게시글을 바탕으로 반려동물의 {texts[0]}을 주제로 반려동물이 사후 하늘나라에서 주인에게 쓰는 안부 인사 편지를 작성해 주세요. 문체는 따뜻하고 감성적인 느낌으로 작성해 주세요. 말투는 반말로 작성해 주세요. 반려동물의 이름은 {pet_name}이고 주인의 이름은 {member_name}입니다. 주인은 {nickname}의 호칭으로 불러주세요."
        letter = generate_letter_answer(memories, letter_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    try:
        title_prompt = f"이 편지 내용의 제목을 10자 이내로 작성해주세요."
        title = generate_letter_answer(letter, title_prompt, OPENAI_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Letter generation failed: {e}"}), 500
    
    if texts[0] == "생일":
        title = "🎂" + title
    elif texts[0] == "기일":    
        title = "🪦" + title

    # GPT로 DreamBooth 프롬프트 추출
    try:
        if texts[0] == "생일":
            prompt_extraction = f"위 내용을 바탕으로 DreamBooth 모델에 적합한 프롬프트를 영어로 아주 짧게 생성하세요. 생성 대상인 반려동물 종류는 {breed}이고 생일케이크 앞에서 행복해하는 반려동물의 모습을 생성하려고 합니다. 'a sks ...' 형식으로 시작해야 합니다."
            dreambooth_prompt = generate_letter_answer(memories, prompt_extraction, OPENAI_API_KEY)
            dreambooth_prompt = "high quality, J_illustration, " + dreambooth_prompt
        elif texts[0]  == "기일":
            prompt_extraction = f"위 내용을 바탕으로 DreamBooth 모델에 적합한 프롬프트를 영어로 아주 짧게 생성하세요. 생성 대상인 반려동물 종류는 {breed}이고 무지개와 별빛을 배경으로 반려동물이 행복하게 지내는 모습을 생성하려고 합니다. 'a sks ...' 형식으로 시작해야 합니다."
            dreambooth_prompt = generate_letter_answer(memories, prompt_extraction, OPENAI_API_KEY)
            dreambooth_prompt = "high quality, J_illustration, " + dreambooth_prompt
    except Exception as e:
        return jsonify({"error": f"Prompt extraction failed: {e}"}), 500
    
    print("🎈 dreambooth_prompt :", dreambooth_prompt)

    try:
        encoded_images = generate_dreambooth(dreambooth_prompt, pet_id)
    except Exception as e:
        return jsonify({"error": f"Dreambooth generation failed: {e}"}), 500
    
    return jsonify({"images": encoded_images, "letter": letter, "title": title})

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

# def run_sam_segmentation(image_rgb, predictor, point):
#     """SAM 모델을 이용해 마스크 예측 수행"""
#     predictor.set_image(image_rgb)
    
#     h, w, _ = image_rgb.shape
#     point = np.array([point])
#     input_label = np.array([1])  # 1: object

#     masks, scores, _ = predictor.predict(
#         point_coords=point,
#         point_labels=input_label,
#         multimask_output=True,
#     )

#     best_mask_index = int(np.argmax(scores))
#     return masks[best_mask_index]

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
    """Convert a processed grayscale image to SVG"""
    height, width = image.shape  # Ensure it's 2D
    dwg = svgwrite.Drawing(svg_path, size=(width, height))
    dwg.attribs['opacity'] = '0.245' 

    for y in range(height):
        for x in range(width):
            if image[y, x] >= threshold:  # Keep high-contrast areas
                dwg.add(dwg.circle(center=(x, y), r=1, fill='#A8C4F7', stroke='none'))
    
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

def enhance_contrast_rgb(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img

@app.route("/stars_run_pidinet", methods=["POST"])
def run_pidinet():
    try:
        data = request.json
        print(f"📥 Received Data: {data}")

        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "No images provided"}), 400

        if isinstance(image_url, str):
            image_url = [image_url]

        stars_download_s3_images(image_urls=image_url, save_folder="./img")

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
            subprocess.run(command, check=True)
            print("✅ PiDiNet 실행 완료!")
            return jsonify({"message": "PiDiNet execution completed"}), 200
        except subprocess.CalledProcessError as e:
            print(f"❌ PiDiNet 실행 실패: {e}")
            return jsonify({"error": "PiDiNet execution failed"}), 500
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@app.route("/stars_process_image", methods=["POST"])
def process_image():
    try:
        data = request.json
        image_url = data.get("image_url")
        point = data.get("point")

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        
        image_name = os.path.basename(image_url)
        image_path = "./img/" + image_name
        
        # # /////////////////////////////////////////////
        
        img = cv2.imread(image_path)
        
        response = requests.get(image_url)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img_yolo = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        enhanced_img = enhance_contrast_rgb(img_yolo)
        
        results = yolo_model(enhanced_img, imgsz=512)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        x = point[0]
        y = point[1]

        selected_box = None
        min_area = float("inf")

        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    selected_box = box
     
        if selected_box is not None:
            sam_results = sam_model(image_path, bboxes=[selected_box])
            mask = sam_results[0].masks.data[0].cpu().numpy()
        else:
            print("점에 해당하는 객체를 찾지 못했습니다.")

        
        # /////////////////////////////////////////////
        
        image, image_rgb = load_and_preprocess_image(image_path)
        # mask = run_sam_segmentation(image_rgb, predictor, point)

        mask_ratio = np.count_nonzero(mask) / mask.size
        max_internal_points = int(mask_ratio * 3000)
        
        contour_points2, internal_points2 = process_segmentation(image, mask, max_internal_points=max_internal_points)
        
        image_path = f"./img_edges/eval_results/imgs_epoch_019/{os.path.splitext(image_name)[0]}.png"
        try:
            image = Image.open(image_path)
            image = np.array(image.convert("L"))
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            return jsonify({"error": "Failed to load edge image"}), 500
        
        masked_image = np.zeros_like(image, dtype=np.float32)
        masked_image[mask > 0] = image[mask > 0]
        processed_image = enhance_contrast(masked_image)

        svg_path = f"{os.path.splitext(image_name)[0]}.svg"
        png_path = f"{os.path.splitext(image_name)[0]}_masked.png"
        
        try:
            image_to_svg(processed_image, svg_path)
            cairosvg.svg2png(url=svg_path, write_to=png_path)
            svg_url = upload_svg_to_s3(BUCKET_NAME, svg_path)
        except Exception as e:
            print(f"❌ 파일 업로드 실패: {e}")
            return jsonify({"error": "Failed to upload SVG file"}), 500

        contour_points1, internal_points1 = sample_pidinet_edges(image, mask, sample_size=max_internal_points)
        internal_points = np.vstack((internal_points1, internal_points2))
        contour_points = np.vstack((contour_points2))

        major_contour = extract_and_sort_centroids(contour_points, n_clusters=20, n_points=5)
        major_internal = extract_and_sort_centroids(internal_points, n_clusters=20, n_points=10)
        major_points = np.vstack((major_contour, major_internal))
        edges = generate_mst_graph(major_points)

        return jsonify({
            "svg_path": svg_url,
            "edges": edges.tolist(),
            "major_points": major_points.astype(int).tolist()
        })
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)