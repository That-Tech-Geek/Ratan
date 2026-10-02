"use client";
import {initializeApp,getApps,getApp} from "firebase/app";
import {getAuth,RecaptchaVerifier,signInWithPhoneNumber,type ConfirmationResult} from "firebase/auth";
const config={apiKey:process.env.NEXT_PUBLIC_FIREBASE_API_KEY,authDomain:process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,projectId:process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,appId:process.env.NEXT_PUBLIC_FIREBASE_APP_ID};
const app=getApps().length?getApp():initializeApp(config);
export const firebaseAuth=getAuth(app);
export function setupRecaptcha(id:string){const w=window as any;if(w.__gyaanRecaptcha)return w.__gyaanRecaptcha as RecaptchaVerifier;const v=new RecaptchaVerifier(firebaseAuth,id,{size:"invisible"});w.__gyaanRecaptcha=v;return v;}
export async function requestOtp(phone:string,id:string):Promise<ConfirmationResult>{const v=setupRecaptcha(id);return signInWithPhoneNumber(firebaseAuth,phone,v);}
export async function currentIdToken(){return firebaseAuth.currentUser?.getIdToken(true)||null;}