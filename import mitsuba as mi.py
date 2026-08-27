import mitsuba as mi
import drjit as dr

class PathIntegratorFinal(mi.ad.integrators.common.ADIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_cameradepth = int(props.get('max_cameradepth', 3))
        self.max_emitterdepth = int(props.get('max_emitterdepth', 3))
        self.light_image_buffer = None
        self.film_size = None
        self.sensor = None
        self.film = None
        self.size = None
        
    def prepare_buffer(self, scene):
        self.sensor: mi.SensorPtr = scene.sensors()[0]
        self.film = self.sensor.film()
        self.size = self.film.crop_size()
        self.film_size = mi.Vector2u(self.size.x, self.size.y)
        self.light_image_buffer = dr.zeros(mi.Float, self.size.x * self.size.y * 3)

    def sample(self,
               mode: dr.ADMode, 
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               active: mi.Bool,
               **kwargs):
        
        ctx_c = mi.BSDFContext(mi.TransportMode.Radiance)
        ctx_e = mi.BSDFContext(mi.TransportMode.Importance) 
        
        L = mi.Spectrum(0.0)
        
        EPW: list[mi.Spectrum] = [mi.Spectrum(0.0) for _ in range(self.max_emitterdepth + 1)]
        EPS: list[mi.SurfaceInteraction3f] = [dr.zeros(mi.SurfaceInteraction3f) for _ in range(self.max_emitterdepth + 1)]
        EPA = [mi.Bool(False) for _ in range(self.max_emitterdepth + 1)]
        EPM = [dr.zeros(mi.Float) for _ in range(self.max_emitterdepth + 1)]
        
        emitter_weight = dr.zeros(mi.Spectrum)
        si = dr.zeros(mi.SurfaceInteraction3f)
        
        # ===================================================================
        # Phase 1. Emitter 샘플링 및 초기화 (s=1 정점 생성)
        # ===================================================================
        emitter_idx, emitter_idx_weight, _ = scene.sample_emitter(sampler.next_1d(), active)
        emitter = dr.gather(mi.EmitterPtr, scene.emitters_dr(), emitter_idx)
        
        is_infinite = mi.has_flag(emitter.flags(), mi.EmitterFlags.Infinite)
        is_delta = mi.has_flag(emitter.flags(), mi.EmitterFlags.Delta)
        active_e = active & ~is_infinite & ~is_delta
        
        ps, pos_weight = emitter.sample_position(0.0, sampler.next_2d(), active_e)
        emitter_weight[active_e] = pos_weight * emitter_idx_weight
        
        si.p[active_e] = ps.p
        si.n[active_e] = ps.n
        si.uv[active_e] = ps.uv
        si.time[active_e] = ps.time
        si.sh_frame[active_e] = mi.Frame3f(ps.n)
        
        wo_local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        wavelenghts, wavelenghts_weight = emitter.sample_wavelengths(si, sampler.next_1d(), active_e)
        si.wavelengths = wavelenghts
        
        si.wi[active_e] = wo_local
        wo_world = si.to_world(wo_local)
        
        EPW[0] = emitter_weight
        EPS[0] = si
        EPA[0] = active_e
        EPM[0] = 0.0
        
        pdf_emit = ps.pdf / emitter_idx_weight # 광원 표면 면적 확률
        emitter_ray = si.spawn_ray(wo_world)
        
        pdf_dir = mi.warp.square_to_cosine_hemisphere_pdf(wo_local)
        safe_pdf_dir = dr.maximum(1e-6, pdf_dir)
        
        path_weight = emitter_weight * wavelenghts_weight * dr.pi 
        path_weight = dr.select(active_e, path_weight, 0.0)
        
        prev_pdf_fwd = safe_pdf_dir * pdf_emit
        prev_n = si.n
        
        # ===================================================================
        # Phase 2. 광원 서브패스 생성 (Light Sub-path)
        # ===================================================================
        for i in range(self.max_emitterdepth):
            si = scene.ray_intersect(emitter_ray)
            active_e &= si.is_valid()
            
            # [FIXED] 1. 샘플링을 '먼저' 수행하여 다음 정점 방향(wo) 확정
            bsdf_sample, bsdf_weight = si.bsdf().sample(ctx_e, si, sampler.next_1d(), sampler.next_2d(), active_e)
            pdf_fwd_next = bsdf_sample.pdf
            
            # [FIXED] 2. 역방향 PDF를 위해 입출사 방향 스왑
            si_rev = dr.zeros(mi.SurfaceInteraction3f)
            si_rev[active_e] = si
            si_rev.wi = bsdf_sample.wo # 나갈 방향에서 들어온다고 가정
            wo_rev = si.wi             # 들어왔던 방향으로 다시 나감
            _, pdf_rev = si.bsdf().eval_pdf(ctx_e, si_rev, wo_rev, active_e)
            
            # [FIXED] 3. 투명체 굴절을 고려한 절대값(dr.abs) 코사인
            cos_curr = dr.abs(dr.dot(si.n, -emitter_ray.d))
            cos_prev = dr.abs(dr.dot(prev_n, emitter_ray.d))
            
            denom = prev_pdf_fwd * cos_curr
            numer = pdf_rev * cos_prev
            
            mis_ratio = dr.select(denom > 1e-6, numer / denom, 0.0)
            mis_ratio *= mis_ratio # [Power Heuristic]
            
            idx = i + 1
            EPM[idx] = mis_ratio * (1.0 + EPM[i]) # [Chain Rule 누적]
            EPS[idx] = si
            EPW[idx] = path_weight
            EPA[idx] = active_e
            
            path_weight *= bsdf_weight
            prev_pdf_fwd = pdf_fwd_next
            prev_n = si.n
            emitter_ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            
        # ===================================================================
        # Phase 3. 광원 정점 -> 카메라(Sensor) 직접 Splatting (t=1 전략)
        # ===================================================================
        for j in range(1, self.max_emitterdepth + 1): 
            si_e = EPS[j]
            weight_e = EPW[j]
            active_e_node = EPA[j]
            mis_light = EPM[j] # EPM은 카메라 방향(C/L) 누적 비율
            
            ds, sensor_val = self.sensor.sample_direction(si_e, sampler.next_2d(), active_e_node)
            active_conn = active_e_node & (ds.pdf > 0.0)
            
            shadow_ray = si_e.spawn_ray_to(ds.p)
            active_conn &= ~scene.ray_test(shadow_ray)
            
            d_vec = ds.d
            dist_sq = dr.maximum(ds.dist * ds.dist, 1e-6)
            cos_e = dr.abs(dr.dot(si_e.n, d_vec))
            cos_c = dr.abs(dr.dot(ds.n, -d_vec))
            
            pdf_sensor_solid = ds.pdf
            # [FIXED] 도착지(Light)의 코사인 cos_e 사용
            pdf_cam_area = dr.select(dist_sq > 1e-6, pdf_sensor_solid * cos_e / dist_sq, 0.0)
            
            wo_local = si_e.to_local(d_vec)
            _, pdf_light_solid = si_e.bsdf().eval_pdf(ctx_e, si_e, wo_local, active_conn)
            # 도착지(Camera)의 코사인 cos_c 사용
            pdf_light_area = dr.select(dist_sq > 1e-6, pdf_light_solid * cos_c / dist_sq, 0.0)
            
            # [FIXED] 카메라 측 확률(C) / 광원 측 확률(L) 방향 맞춤
            ratio_cam_side = dr.select(pdf_light_area > 1e-6, pdf_cam_area / pdf_light_area, 0.0)
            ratio_cam_side *= ratio_cam_side # [Power Heuristic]
            
            # C/L 비율이므로 C/L 누적합인 mis_light(EPM)과 Chain 결합
            w_cam_side = dr.select(pdf_light_area > 1e-6, ratio_cam_side * (1.0 + mis_light), 0.0)
            
            mis_weight = 1.0 / (1.0 + w_cam_side)
            mis_weight = dr.select(active_conn & dr.isfinite(mis_weight), mis_weight, 0.0)
            
            bsdf_val = si_e.bsdf().eval(ctx_e, si_e, wo_local, active_conn)
            L_connect = weight_e * sensor_val * bsdf_val
            L_connect = dr.clip(L_connect, 0.0, 1000.0)
            L_connect = dr.select(active_conn, L_connect * mis_weight, 0.0)
            
            width_f = mi.Float(self.film_size.x)
            height_f = mi.Float(self.film_size.y)
            pos_x = dr.clip(dr.floor(ds.uv.x), 0.0, width_f - 1.0)
            pos_y = dr.clip(dr.floor(ds.uv.y), 0.0, height_f - 1.0)
            pixel_idx = (mi.UInt32(pos_y) * self.film_size.x + mi.UInt32(pos_x)) * 3
            
            dr.scatter_add(target=self.light_image_buffer, value=L_connect[0], index=pixel_idx + 0, active=active_conn)
            dr.scatter_add(target=self.light_image_buffer, value=L_connect[1], index=pixel_idx + 1, active=active_conn)
            dr.scatter_add(target=self.light_image_buffer, value=L_connect[2], index=pixel_idx + 2, active=active_conn)
            
        # ===================================================================
        # Phase 4. 카메라 서브패스 및 양방향 연결
        # ===================================================================
        beta_c = mi.Spectrum(1.0)
        active_c = active
        mis_cam = mi.Float(0.0)
        
        prev_pdf_fwd_c = mi.Float(1.0) 
        prev_n_c = dr.zeros(mi.Vector3f)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_delta = mi.Bool(True)
        
        si = scene.ray_intersect(ray)
        
        # 첫 번째 정점 직접 조명 (s=0, t=2) - 대안 전략이 없으므로 MIS 1.0
        Le_direct = si.emitter(scene).eval(si)
        L += beta_c * Le_direct 
        
        for i in range(self.max_cameradepth):
            active_c &= si.is_valid()
            bsdf = si.bsdf()
            
            # --- [A] 우연히 광원을 맞춘 경우 (Implicit Emitter Hit, s=0) ---
            if i > 0:
                hit_emitter_ptr = si.emitter(scene)
                active_hit = active_c & dr.neq(hit_emitter_ptr, None)
                
                ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
                pdf_nee = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
                
                w_nee = dr.select(prev_pdf_fwd_c > 1e-6, pdf_nee / prev_pdf_fwd_c, 0.0)
                w_nee *= w_nee # [Power Heuristic]
                
                # [FIXED] 질문자님의 완벽한 Chain Rule 수식 유지!
                denom_w = 1.0 + w_nee * (1.0 + mis_cam) 
                
                mis_weight_bsdf = dr.select(denom_w > 1e-6, 1.0 / denom_w, 0.0)
                mis_weight_bsdf = dr.select(dr.isfinite(mis_weight_bsdf), mis_weight_bsdf, 0.0)
                
                Le_bsdf_hit = hit_emitter_ptr.eval(si, active_hit)
                Le_bsdf_hit = dr.clip(Le_bsdf_hit, 0.0, 1000.0)
                L += dr.select(active_hit, beta_c * Le_bsdf_hit * mis_weight_bsdf, 0.0)

            # --- [B] 순정 양방향 연결 루프 (s=1 포함, s >= 1) ---
            for j in range(0, self.max_emitterdepth + 1): 
                si_e = EPS[j]
                weight_e = EPW[j]
                active_e_node = EPA[j]
                
                active_conn = active_c & active_e_node
                d_conn = si_e.p - si.p
                dist_sq = dr.maximum(dr.squared_norm(d_conn), 1e-6)
                dist = dr.sqrt(dist_sq)
                wo_c = d_conn / dist
                wo_e = -wo_c
                
                shadow_ray = si.spawn_ray_to(si_e.p)
                is_delta_cam = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)
                is_delta_emit = mi.Bool(False)
                if j > 0:
                    is_delta_emit = mi.has_flag(si_e.bsdf().flags(), mi.BSDFFlags.Delta)
                    
                visible = active_conn & ~scene.ray_test(shadow_ray) & ~is_delta_cam & ~is_delta_emit
                
                cos_c = dr.abs(dr.dot(si.n, wo_c))
                cos_e = dr.abs(dr.dot(si_e.n, wo_e))
                
                _, pdf_c2l = bsdf.eval_pdf(ctx_c, si, si.to_local(wo_c), visible)
                p_A_c2l = pdf_c2l * cos_e / dist_sq
                
                p_A_l2c = mi.Float(0.0)
                if j == 0:
                    # j=0 (s=1 전략): EPS[0]의 원래 면적 확률 그대로 사용
                    p_A_l2c = pdf_emit 
                else:
                    _, pdf_l2c_solid = si_e.bsdf().eval_pdf(ctx_e, si_e, si_e.to_local(wo_e), visible)
                    p_A_l2c = pdf_l2c_solid * cos_c / dist_sq
                
                # [FIXED] 비율 및 방향 맞춤 (Chain Rule)
                ratio_towards_light = dr.select(p_A_l2c > 1e-6, p_A_c2l / p_A_l2c, 0.0)
                ratio_towards_light *= ratio_towards_light # [Power Heuristic]
                
                ratio_towards_camera = dr.select(p_A_c2l > 1e-6, p_A_l2c / p_A_c2l, 0.0)
                ratio_towards_camera *= ratio_towards_camera # [Power Heuristic]
                
                # 빛 방향(C/L)은 EPM(C/L)과, 카메라 방향(L/C)은 mis_cam(L/C)과 짝을 이룸
                w_light_side = ratio_towards_light * (1.0 + EPM[j])
                w_cam_side = ratio_towards_camera * (1.0 + mis_cam)
                
                mis_weight = 1.0 / (1.0 + w_cam_side + w_light_side)
                mis_weight = dr.select(visible & dr.isfinite(mis_weight), mis_weight, 0.0)
                
                f_cam = bsdf.eval(ctx_c, si, si.to_local(wo_c), visible)
                contrib = mi.Spectrum(0.0)
                
                if j == 0:
                    si_e.wi = si_e.to_local(wo_e)
                    Le_connect = emitter.eval(si_e, visible)
                    G = cos_e / dist_sq
                    contrib = beta_c * Le_connect * weight_e * G * f_cam
                else:
                    f_emit = si_e.bsdf().eval(ctx_e, si_e, si_e.to_local(wo_e), visible)
                    G = 1.0 / dist_sq
                    contrib = beta_c * f_cam * G * f_emit * weight_e 
                
                contrib = dr.clip(contrib, 0.0, 1000.0)
                L += dr.select(visible, contrib * mis_weight, 0.0) 

            # --- [C] 카메라 경로 연장 및 역방향 누적 ---
            # 1. 방향 결정
            bsdf_sample, bsdf_weight = bsdf.sample(ctx_c, si, sampler.next_1d(), sampler.next_2d(), active_c)
            
            # 2. 역방향 계산 (입출사 스왑)
            si_rev_c = dr.zeros(mi.SurfaceInteraction3f)
            si_rev_c[active_c] = si
            si_rev_c.wi = bsdf_sample.wo 
            wo_rev_c = si.wi             
            _, pdf_rev_c = si.bsdf().eval_pdf(ctx_c, si_rev_c, wo_rev_c, active_c)
            
            # 3. 절대값 코사인 & 면적 비율 계산
            cos_curr_c = dr.abs(dr.dot(si.n, -ray.d))
            cos_prev_c = dr.abs(dr.dot(prev_n_c, ray.d))
            
            denom_c = prev_pdf_fwd_c * cos_curr_c
            numer_c = pdf_rev_c * cos_prev_c
            
            ratio = dr.select(denom_c > 1e-6, numer_c / denom_c, 0.0)
            ratio *= ratio # [Power Heuristic]
            
            # 4. Chain Rule 누적
            mis_cam = ratio * (1.0 + mis_cam) 
            
            # 5. 다음 루프를 위한 상태 갱신
            prev_pdf_fwd_c = bsdf_sample.pdf
            prev_n_c = si.n
            prev_si = si
            beta_c *= bsdf_weight
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            
        return L, active, [], None