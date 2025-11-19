"""
학습 루프 모듈
train step / eval step 분리, epoch별 로그, best model checkpoint 저장, early stopping 지원
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

from .metrics import calculate_all_metrics
try:
    from ..utils.logger import setup_logger
except ImportError:
    from utils.logger import setup_logger


def train_step(model, images, labels, criterion, optimizer, device):
    """
    단일 배치 학습 스텝
    
    Args:
        model: 학습할 모델
        images: 입력 이미지
        labels: 레이블
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        
    Returns:
        loss: 손실값
        outputs: 모델 출력
    """
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    
    # 순전파
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 역전파
    loss.backward()
    optimizer.step()
    
    return loss.item(), outputs


def eval_step(model, images, labels, criterion, device):
    """
    단일 배치 평가 스텝
    
    Args:
        model: 평가할 모델
        images: 입력 이미지
        labels: 레이블
        criterion: 손실 함수
        device: 디바이스
        
    Returns:
        loss: 손실값
        outputs: 모델 출력
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    return loss.item(), outputs


def train_epoch(model, dataloader, criterion, optimizer, device, 
                class_names=None, logger=None):
    """
    한 에포크 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        class_names: 클래스 이름 리스트 (선택)
        logger: 로거 객체 (선택)
        
    Returns:
        metrics: 학습 메트릭 딕셔너리
    """
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        # 학습 스텝
        loss, outputs = train_step(model, images, labels, criterion, optimizer, device)
        
        running_loss += loss
        all_outputs.append(outputs)
        all_labels.append(labels)
        
        # 진행률 표시
        pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    # 전체 배치의 출력과 레이블 수집
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 메트릭 계산
    metrics = calculate_all_metrics(all_outputs, all_labels, class_names=class_names)
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, 
                  class_names=None, logger=None):
    """
    한 에포크 검증
    
    Args:
        model: 검증할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        class_names: 클래스 이름 리스트 (선택)
        logger: 로거 객체 (선택)
        
    Returns:
        metrics: 검증 메트릭 딕셔너리
    """
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            # 평가 스텝
            loss, outputs = eval_step(model, images, labels, criterion, device)
            
            running_loss += loss
            all_outputs.append(outputs)
            all_labels.append(labels)
            
            # 진행률 표시
            pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    # 전체 배치의 출력과 레이블 수집
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 메트릭 계산
    metrics = calculate_all_metrics(all_outputs, all_labels, class_names=class_names)
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


class EarlyStopping:
    """
    Early Stopping 구현
    검증 손실이 개선되지 않으면 학습 조기 종료
    """
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience: 개선 없이 기다릴 에포크 수
            min_delta: 최소 개선량
            mode: 'min' (손실 최소화) 또는 'max' (메트릭 최대화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Args:
            score: 현재 점수 (낮을수록 좋으면 mode='min', 높을수록 좋으면 mode='max')
        
        Returns:
            early_stop: 조기 종료 여부
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current, best):
        """현재 점수가 더 좋은지 확인"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, class_names=None, save_dir='experiments/checkpoints',
                log_dir='experiments/logs', model_name='model', 
                early_stopping=None, scheduler_name='cosine_annealing'):
    """
    전체 학습 루프
    
    Args:
        model: 학습할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        num_epochs: 총 에포크 수
        device: 디바이스
        class_names: 클래스 이름 리스트
        save_dir: 모델 저장 디렉토리
        log_dir: 로그 저장 디렉토리
        model_name: 모델 이름
        early_stopping: EarlyStopping 객체 (선택)
        scheduler_name: 스케줄러 이름
        
    Returns:
        history: 학습 히스토리 딕셔너리
    """
    # 디렉토리 생성
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(f'{model_name}_train', log_dir=str(log_dir))
    
    # 학습 히스토리
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'best_val_accuracy': 0.0
    }
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    logger.info("=" * 70)
    logger.info(f"학습 시작: {model_name}")
    logger.info(f"총 에포크: {num_epochs}")
    logger.info(f"디바이스: {device}")
    logger.info("=" * 70)
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*70}")
        
        # 학습
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                   device, class_names=class_names, logger=logger)
        
        # 검증
        val_metrics = validate_epoch(model, val_loader, criterion, device,
                                    class_names=class_names, logger=logger)
        
        # 학습률 스케줄러 업데이트
        if scheduler_name.lower() == 'cosine_annealing':
            scheduler.step()
        elif scheduler_name.lower() == 'reduce_lr_on_plateau':
            scheduler.step(val_metrics['loss'])
        
        # 히스토리 업데이트
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 현재 학습률
        current_lr = optimizer.param_groups[0]['lr']
        
        # 로그 출력
        logger.info(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Best model 저장
        is_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_accuracy = val_metrics['accuracy']
            history['best_epoch'] = epoch
            history['best_val_loss'] = best_val_loss
            history['best_val_accuracy'] = best_val_accuracy
            is_best = True
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': history
        }
        
        # 최신 모델 저장
        latest_path = save_dir / f'{model_name}_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Best model 저장
        if is_best:
            best_path = save_dir / f'{model_name}_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model 저장: {best_path} "
                       f"(Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_accuracy:.4f})")
        
        # Early Stopping 체크
        if early_stopping is not None:
            if early_stopping(val_metrics['loss']):
                logger.info(f"\n⚠️  Early Stopping triggered at epoch {epoch}")
                logger.info(f"Best epoch: {history['best_epoch']}")
                logger.info(f"Best Val Loss: {history['best_val_loss']:.4f}")
                logger.info(f"Best Val Accuracy: {history['best_val_accuracy']:.4f}")
                break
    
    # 학습 히스토리 저장
    history_path = log_dir / f'{model_name}_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"\n학습 히스토리 저장: {history_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("학습 완료!")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info(f"Best Val Loss: {history['best_val_loss']:.4f}")
    logger.info(f"Best Val Accuracy: {history['best_val_accuracy']:.4f}")
    logger.info("=" * 70)
    
    return history

