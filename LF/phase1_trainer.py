"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from itertools import chain

import torch
import torch.nn.functional as F

from base.trainer import BaseTrainer, cyclize
import base.utils as utils

import cv2
import json

def to_batch(batch):
    in_batch = {
        "ref_imgs": batch["ref_imgs"].cuda(),
        "ref_fids": batch["ref_fids"].cuda(),
        "ref_decs": batch["ref_decs"].cuda(),
        "trg_fids": batch["trg_fids"].cuda(),
        "trg_decs": batch["trg_decs"],
        "src_imgs": batch["src_imgs"].cuda(),
        "phase": "comb"
    }
    return in_batch


class LF1Trainer(BaseTrainer):
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, cfg, use_ddp):
        super().__init__(gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                         writer, logger, cfg, use_ddp)

        self.to_batch = to_batch

    def train(self, loader, val_loaders, max_step=100000):

        ## 레퍼 이미지를 보고 
        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm",
                                     "ac", "ac_gen")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target", "ac_acc", "ac_gen_acc")

        self.clear_losses()

        self.logger.info("Start training ...")

        for batch in cyclize(loader):
            epoch = self.step // len(loader)
            if self.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)



            ## primals 
            with open('./data/kor/primals.json') as f:
                primals_dict = json.load(f)
            ## 삭제해도 됨
            ## decom포넌트 
            with open('./data/kor/train_chars.json') as f:
                train_chars = json.load(f)
            ## 삭제해도 됨
            
            ## 텐서이미지
            ref_imgs = batch["ref_imgs"].cuda()
            ## ref_fids는 폰트 아이디
            ref_fids = batch["ref_fids"].cuda()
            ## 문자열 분해 번호를 일렬로 나열
            ref_decs = batch["ref_decs"].cuda()
            # primals_dict를 일렬로 나열
            # 땟텬림이면
            # 2 ㄷ , 2 ㄷ , 14 ㅏ , 23 ㅣ, 6 ㅅ
            # 11 ㅌ , 17 ㅕ, 1 ㄴ
            # 3 ㄹ , 23 ㅣ, 4 ㅁ
            # tensor([ 2,  2, 14, 23,  6, 11, 17,  1,  3, 23,  4], device='cuda:0')
            
            ## 디버깅 용도를 위한 래퍼 이미지 저장
            # temp = ref_imgs.detach().cpu().numpy()
            # temp = temp.transpose(0,2,3,1)
            # temp = temp*255
            # b_img,_,_,_ = temp.shape
            # for i in range(0,b_img):
            #     cv2.imwrite("./Tr/ref_"+str(i)+".png",temp[i,:,:,:])
            ## 삭제 해도 됨
            
            

                        
        
            trg_imgs = batch["trg_imgs"].cuda()
            trg_fids = batch["trg_fids"].cuda()            
            ## trg_cids는 캐릭터 아이디 -> train_chars.json을 참조함
            trg_cids = batch["trg_cids"].cuda()     
            ## trg_decs는 primals.json를 참조함
            trg_decs = batch["trg_decs"]            
            src_imgs = batch["src_imgs"].cuda()
            
            
            # list(decomposition.keys())[655]
            
            ## 디버깅 용도를 위한 래퍼 이미지 저장
            temp = trg_imgs.detach().cpu().numpy()
            temp = temp.transpose(0,2,3,1)
            temp = temp*255
            b_img,_,_,_ = temp.shape
            for i in range(0,b_img):
                cv2.imwrite("./Tr/target_"+str(i)+".png",temp[i,:,:,:])
            
            
            
            
            
            
            
            

            
            # ## 삭제 해도 됨   
            # ## 디버깅 용도를 위한 래퍼 이미지 저장
            # temp = src_imgs.detach().cpu().numpy()
            # temp = temp.transpose(0,2,3,1)
            # temp = temp*255
            # b_img,_,_,_ = temp.shape
            # for i in range(0,b_img):
            #     cv2.imwrite("./Tr/src_"+str(i)+".png",temp[i,:,:,:])
            # ## 삭제 해도 됨   
            
            
            
            
            
            
            
            
            

            B = len(trg_imgs)
            stats.updates({
                "B_style": ref_imgs.size(0),
                "B_target": B
            })
            ## 타겟이미지 B = 배치사이즈
            ## B_style = 참조할 래프 이미지 배치 사이즈
            


            ## sc_feats
            ## torch.Size([44, 256, 16, 16])
            ## 참조 이미지를 인코더함            
            ## 참조이미지의 스타일별 임베딩 벡터를 가져옴
            sc_feats = self.gen.encode_write_comb(ref_fids, ref_decs, ref_imgs)
            
            
            
            
            gen_imgs = self.gen.read_decode(trg_fids, trg_decs, src_imgs, phase="comb")
            # 타겟의 폰트 아이디, 타겟의 문자 컴포넌트, 소스이미를 가지고
            # 일단 한번 제너레이션을 한번 해보겠다는 뜻임
            # trg_decs = torch.Size([5, 1, 128, 128])
            # [[4, 14, 1], [0, 16, 0], [0, 0, 14, 0, 0], [5, 16], [5, 5, 14, 23]]
            
            

            ## self.cfg['fm_layers'] = 'all'로 되어있음
            ## 진짜(타겟) 이미지를 판별기에 넣고 나온 결과
            real_font, real_uni, *real_feats = self.disc(
                trg_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )
            # 판별자에서 나온 결과물
            # real_font = torch.Size([5, 1, 1, 1]) 폰트 종류 임베딩 숫자 들어있음
            # real_uni = 이글자가 무엇인지 임베딩 벡터            
            # real_feats = 판별자의 아키텍쳐(CNN,레즈블럭)에서 나온 값(특징들)
            



            ## 생성된 이미지를 넣고 결과를 받음
            fake_font, fake_uni = self.disc(gen_imgs.detach(), trg_fids, trg_cids)
            # fake_font = 만들어진 이미지가 어느 폰트에 속하는지 
            # fake_uni = 만들어진 이미지가 어느 글자에 속하는지의 결과값
            
            
            ## 진짜 이미지,가짜 이미지를 넣은 결과값들의 loss를 add함
            ## 제너레이터와, 판별자간의 loss를 구함 (l1 loss)
            self.add_gan_d_loss([real_font, real_uni], [fake_font, fake_uni])
            
            ## 판별자 백워드
            self.d_optim.zero_grad()
            ## 제너레이터 얼리고 백워드 수행
            self.d_backward()
            self.d_optim.step()

            
            
            
            
            
            ##
            fake_font, fake_uni, *fake_feats = self.disc(
                gen_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )
            
            
            ## 평균
            self.add_gan_g_loss(fake_font, fake_uni)

            
            # FM 모듈도 L1 로스사용
            self.add_fm_loss(real_feats, fake_feats)
            
            

            def racc(x):
                return (x > 0.).float().mean().item()

            def facc(x):
                return (x < 0.).float().mean().item()

            discs.updates({
                "real_font": real_font.mean().item(),
                "real_uni": real_uni.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_uni": fake_uni.mean().item(),

                'real_font_acc': racc(real_font),
                'real_uni_acc': racc(real_uni),
                'fake_font_acc': facc(fake_font),
                'fake_uni_acc': facc(fake_uni)
            }, B)

            
            ## 생성과 타겟이미지간의 loss
            self.add_pixel_loss(gen_imgs, trg_imgs)

            
            self.g_optim.zero_grad()

            
            ## acu_clf보조 네트워크 여기서 사용
            self.add_ac_losses_and_update_stats(
                sc_feats, ref_decs, gen_imgs, trg_decs, stats
            )
            
            ## aux_clf 보조 
            self.ac_optim.zero_grad()
            self.ac_backward()
            self.ac_optim.step()


            ## gan 백워드
            self.g_backward()
            self.g_optim.step()


            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)  # accum loss stats


            # EMA https://study-grow.tistory.com/entry/gema-EMA-%EA%B5%AC%ED%95%98%EB%8A%94-%EA%B3%B5%EC%8B%9D
            # 최근에 학습한 가중치일수록 더 많이 되도록 함
            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(batch)

            torch.cuda.synchronize()

            if self.cfg.rank == 0:
                if self.step % self.cfg.tb_freq == 0:
                    self.plot(losses, discs, stats)

                if self.step % self.cfg.print_freq == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_cached() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                    nrow = len(trg_imgs)
                    grid = utils.make_comparable_grid(src_imgs.detach().cpu(),
                                                      trg_imgs.detach().cpu(),
                                                      gen_imgs.detach().cpu(),
                                                      nrow=nrow)
                    self.writer.add_image("last", grid)

                if self.step > 0 and self.step % self.cfg.val_freq == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))

                    if not self.is_bn_gen:
                        self.sync_g_ema(batch)

                    for _key, _loader in val_loaders.items():
                        n_row = _loader.dataset.n_gen
                        self.infer_save_img(_loader, tag=_key, n_row=n_row)

                    self.save(self.cfg.save, self.cfg.get('save_freq', self.cfg.val_freq))
            else:
                pass

            if self.step >= max_step:
                break

            self.step += 1

        self.logger.info("Iteration finished.")

    def infer_ac(self, sc_feats, comp_ids):
        ## 컴포넌트를 예측하는 -> 보조 분류기
        ## 정답 comp_ids tensor([ 2, 14, 23,  0,  3, 16,  6,  6, 12, 20,  6], device='cuda:0')
        ## aux_out = torch.Size([11, 24])
        
        aux_out = self.aux_clf(sc_feats)
        loss = F.cross_entropy(aux_out, comp_ids)
        acc = utils.accuracy(aux_out, comp_ids)
        return loss, acc

    def add_ac_losses_and_update_stats(self, in_sc_feats, in_decs, gen_imgs, trg_decs, stats):
        
        ## 
        # torch.Size([11, 256, 16, 16])
        
        loss, acc = self.infer_ac(in_sc_feats, in_decs)
        
        
        self.ac_losses['ac'] = loss * self.cfg['ac_w']
        stats.ac_acc.update(acc, in_decs.numel())

        trg_comp_lens = torch.LongTensor([*map(len, trg_decs)]).cuda()
        trg_comp_ids = torch.LongTensor([*chain(*trg_decs)]).cuda()
        generated = gen_imgs.repeat_interleave(trg_comp_lens, dim=0)

        feats = self.gen_ema.comp_enc(generated, trg_comp_ids)
        gen_comp_feats = feats["last"]

        loss, acc = self.infer_ac(gen_comp_feats, trg_comp_ids)
        stats.ac_gen_acc.update(acc, trg_comp_ids.numel())
        self.frozen_ac_losses['ac_gen'] = loss * self.cfg['ac_gen_w']

    def log(self, L, D, S):
        self.logger.info(
            f"Step {self.step:7d}\n"
            f"{'|D':<12} {L.disc.avg:7.3f} {'|G':<12} {L.gen.avg:7.3f} {'|FM':<12} {L.fm.avg:7.3f} {'|R_font':<12} {D.real_font_acc.avg:7.3f} {'|F_font':<12} {D.fake_font_acc.avg:7.3f} {'|R_uni':<12} {D.real_uni_acc.avg:7.3f} {'|F_uni':<12} {D.fake_uni_acc.avg:7.3f}\n"
        )

## 모델 저장
# import torch.onnx
# torch.onnx.export(self.disc,(trg_imgs, trg_fids, trg_cids, self.cfg['fm_layers']),'./temp.onnx',export_params=True,opset_version=12)

